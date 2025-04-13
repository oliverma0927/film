import sys, os, traceback, cv2, rawpy, numpy as np
from PIL import Image, ImageCms
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSlider, QShortcut, QGridLayout,
    QDoubleSpinBox, QCheckBox
)
from PyQt5.QtCore import Qt, QRect, QTimer, QEvent
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QKeySequence

import tifffile  # 用于真正的16位TIFF写出

PROPHOTO_ICC = "ProPhoto.icc"   # 请放在脚本同一目录
WIDEGAMUT_ICC= "WideGamut.icc"  # 同理

def read_raw_widegamut(file_path):
    """
    使用 rawpy 读取 RAW 并解码到 WideGamut(16bit)
    """
    with rawpy.imread(file_path) as raw:
        rgb16 = raw.postprocess(
            use_auto_wb=False,
            gamma=(1.0,1.0),
            no_auto_bright=True,
            output_color=rawpy.ColorSpace.Wide,
            output_bps=16
        )
    return rgb16

def read_tiff_anydepth(file_path):
    """
    用 PIL 读TIFF(可能8或16)，统一变成 16位uint
    """
    with Image.open(file_path) as pil_img:
        if pil_img.mode!="RGB":
            pil_img=pil_img.convert("RGB")
        arr= np.array(pil_img)
        if arr.dtype== np.uint8:
            arr_16= (arr.astype(np.uint16)<<8)
        else:
            arr_16= arr.astype(np.uint16)
        return arr_16

def invert_negative(rgb_float):
    """负片 -> 反相"""
    return 1.0 - rgb_float

def float_to_uint16(rgb_float):
    """
    将[0,1] float32 -> [0,65535] uint16
    返回 numpy uint16数组
    """
    return np.clip(rgb_float*65535, 0,65535).astype(np.uint16)

def convert_to_prophoto_pil(pil_img_16):
    """
    假设输入近似 sRGB，转成ProPhoto
    """
    if not os.path.exists(PROPHOTO_ICC):
        raise FileNotFoundError("找不到 ProPhoto ICC:"+ PROPHOTO_ICC)
    srgb=ImageCms.createProfile("sRGB")
    prophoto= ImageCms.ImageCmsProfile(PROPHOTO_ICC)
    transform= ImageCms.buildTransform(srgb, prophoto, "RGB","RGB", renderingIntent=0)
    return ImageCms.applyTransform(pil_img_16, transform)

def downsample_float(img_float, max_dim=1000):
    """用cv2缩放 float图,适合预览"""
    h,w,c= img_float.shape
    longest=max(h,w)
    if longest<=max_dim:
        return img_float.copy()
    scale= max_dim/ float(longest)
    new_w= int(round(w*scale))
    new_h= int(round(h*scale))
    resized= cv2.resize(img_float,(new_w,new_h), interpolation=cv2.INTER_AREA)
    return resized

def channel_balance_auto_levels(rgb_float,low_cut=0.001, high_cut=0.001):
    out= np.empty_like(rgb_float)
    for ch in range(3):
        ch_data= rgb_float[...,ch].ravel()
        low_val= np.percentile(ch_data, low_cut*100)
        high_val= np.percentile(ch_data,(1 - high_cut)*100)
        if high_val-low_val<1e-6:
            out[...,ch]= rgb_float[...,ch]
        else:
            tmp= (rgb_float[...,ch]-low_val)/(high_val-low_val)
            out[...,ch]= tmp
    return out

def compute_autolevel_params(region, low_cut=0.001, high_cut=0.001):
    """
    仅计算 autolevel所需的 (low_val,high_val) 每通道
    """
    params=[]
    for ch in range(3):
        ch_data= region[...,ch].ravel()
        low_val= np.percentile(ch_data, low_cut*100)
        high_val= np.percentile(ch_data,(1 - high_cut)*100)
        params.append((low_val, high_val))
    return params

def apply_autolevels_with_params(img, params, eps=1e-5):
    """
    用已计算的 params 对图像进行线性拉伸
    """
    out= np.empty_like(img)
    for ch in range(3):
        low_val,high_val= params[ch]
        denom= (high_val-low_val) if (high_val>low_val) else eps
        out[...,ch]= (img[...,ch]-low_val)/denom
    return out

def apply_color_balance(rgb_float, temp, tint):
    """
    简易色温/色调
    """
    red_gain= 1 + (temp/100.0)
    blue_gain=1 - (temp/100.0)
    green_gain=1 + (tint/100.0)
    out= rgb_float.copy()
    out[...,0]*= red_gain
    out[...,1]*= green_gain
    out[...,2]*= blue_gain
    return out

def apply_manual_white_balance(rgb_float, gains):
    """手动白平衡增益 (gain_R, gain_B)"""
    gain_R, gain_B= gains
    out= rgb_float.copy()
    out[...,0]*= gain_R
    out[...,2]*= gain_B
    return out

def apply_exposure(rgb_float, exp_val):
    """
    线性曝光: factor=1+exp_val/2.0
    """
    factor= 1 + exp_val/2.0
    return rgb_float*factor

def apply_gamma(rgb_float, gamma_val):
    """
    gamma校正: out = in^(gamma_val)
    """
    return np.power(rgb_float, gamma_val)

def apply_curve(rgb_float, curve_val):
    """
    简易曲线: -50..50 -> -0.5..0.5
    out=0.5 + (in-0.5)*(1+s)
    """
    s= curve_val/100.0
    if abs(s)<1e-6:
        return rgb_float
    out= 0.5+(rgb_float-0.5)*(1.0+s)
    return out

def apply_saturation(rgb_float, sat_val):
    """
    饱和度: sat_val -50..50 => -0.5..+0.5
    lum=0.299r+0.587g+0.114b
    out= lum + (in-lum)*(1+s)
    """
    s= sat_val/100.0
    if abs(s)<1e-6:
        return rgb_float
    lum= 0.299*rgb_float[...,0]+ 0.587*rgb_float[...,1]+ 0.114*rgb_float[...,2]
    lum= lum[...,np.newaxis]
    out= lum + (rgb_float-lum)*(1.0+s)
    return out

def pipeline_process(base, temp,tint, exp_val, gamma_val, curve_val, sat_val, manual_wb=None):
    out= base.copy()
    if manual_wb is not None:
        out= apply_manual_white_balance(out, manual_wb)
    else:
        out= apply_color_balance(out, temp, tint)
    out= apply_exposure(out, exp_val)
    out= apply_gamma(out, gamma_val)
    out= apply_curve(out, curve_val)
    out= apply_saturation(out, sat_val)
    return np.clip(out,0,1)

def convert_to_srgb(np_img):
    """
    将 widegamut float图转成 sRGB float图,需 widegamut.icc
    """
    try:
        pil_img= Image.fromarray( (np.clip(np_img*255,0,255)).astype(np.uint8),"RGB")
        wide_profile= ImageCms.ImageCmsProfile(WIDEGAMUT_ICC)
        srgb_profile= ImageCms.createProfile("sRGB")
        transform= ImageCms.buildTransformFromOpenProfiles(wide_profile, srgb_profile,"RGB","RGB")
        pil_srgb= ImageCms.applyTransform(pil_img, transform)
        srgb_arr= np.array(pil_srgb).astype(np.float32)/255.0
        return srgb_arr
    except Exception as e:
        print("转换到 sRGB 失败：", e)
        return np_img

def crop_image_center(img, black_pct, white_pct):
    """
    中心裁切：上/左裁 black_pct%, 下/右裁 white_pct%
    """
    h,w,_=img.shape
    lw= int(w*(black_pct/100.0))
    rw= int(w*(white_pct/100.0))
    th= int(h*(black_pct/100.0))
    bh= int(h*(white_pct/100.0))
    return img[th:h-bh, lw:w-rw, :]

class CropAndRotateLabel(QLabel):
    """
    支持鼠标拖拽画选区(红框)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.startPos=None
        self.endPos=None
        self.dragging=False
        self._cropRect=None
    def mousePressEvent(self,event):
        if event.button()==Qt.LeftButton:
            self.dragging=True
            self.startPos=event.pos()
            self.endPos=event.pos()
            self.update()
    def mouseMoveEvent(self,event):
        if self.dragging:
            self.endPos= event.pos()
            self.update()
    def mouseReleaseEvent(self,event):
        if self.dragging and event.button()==Qt.LeftButton:
            self.dragging=False
            self.endPos= event.pos()
            rect= self.getRect()
            if rect.width()>5 and rect.height()>5:
                self._cropRect= (rect.x(),rect.y(), rect.width(),rect.height())
            else:
                self._cropRect=None
            self.update()
    def paintEvent(self,event):
        super().paintEvent(event)
        if self._cropRect:
            p= QPainter(self)
            x,y,w,h= self._cropRect
            p.setPen(QPen(Qt.red,2,Qt.SolidLine))
            p.drawRect(x,y,w,h)
    def getRect(self):
        if (not self.startPos) or (not self.endPos):
            return QRect()
        x1,y1= self.startPos.x(), self.startPos.y()
        x2,y2= self.endPos.x(), self.endPos.y()
        return QRect(min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1))
    def getSelectedRect(self):
        return self._cropRect
    def clearCropRect(self):
        self._cropRect=None
        self.update()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("选区+黑白场+Gamma+曝光+保存16位TIFF(tifffile)")
        self.resize(1300,800)
        w=QWidget(); self.setCentralWidget(w)
        self.main_layout= QVBoxLayout(w)

        # 按钮行
        row_btn= QHBoxLayout()
        self.btn_open= QPushButton("打开")
        self.btn_save= QPushButton("保存")
        self.btn_confirm=QPushButton("确认选区")
        self.btn_cancel=QPushButton("取消选区")
        self.btn_rotate=QPushButton("旋转90°")
        self.btn_flip=QPushButton("垂直翻转")
        self.btn_select_wb=QPushButton("选择中性灰点")
        self.btn_clear_wb=QPushButton("清除WB")
        self.btn_reset=QPushButton("重置")

        row_btn.addWidget(self.btn_open)
        row_btn.addWidget(self.btn_save)
        row_btn.addWidget(self.btn_confirm)
        row_btn.addWidget(self.btn_cancel)
        row_btn.addWidget(self.btn_rotate)
        row_btn.addWidget(self.btn_flip)
        row_btn.addWidget(self.btn_select_wb)
        row_btn.addWidget(self.btn_clear_wb)
        row_btn.addWidget(self.btn_reset)

        self.btn_open.clicked.connect(self.on_open)
        self.btn_save.clicked.connect(self.on_save)
        self.btn_confirm.clicked.connect(self.on_confirm_selection)
        self.btn_cancel.clicked.connect(self.on_clear_selection)
        self.btn_rotate.clicked.connect(self.on_rotate)
        self.btn_flip.clicked.connect(self.on_flip_vertical)
        self.btn_select_wb.clicked.connect(self.on_select_wb)
        self.btn_clear_wb.clicked.connect(self.on_clear_wb)
        self.btn_reset.clicked.connect(self.on_reset)

        self.main_layout.addLayout(row_btn)

        # 黑白场
        row2= QHBoxLayout()
        self.checkbox_crop=QCheckBox("黑白场")
        self.checkbox_crop.setChecked(False)
        self.spin_black=QDoubleSpinBox()
        self.spin_black.setRange(0,50)
        self.spin_black.setSingleStep(0.001)
        self.spin_black.setValue(0.005)
        self.spin_white=QDoubleSpinBox()
        self.spin_white.setRange(0,50)
        self.spin_white.setSingleStep(0.001)
        self.spin_white.setValue(0.005)
        self.spin_black.valueChanged.connect(self.update_preview)
        self.spin_white.valueChanged.connect(self.update_preview)

        row2.addWidget(self.checkbox_crop)
        row2.addWidget(QLabel("黑场%"))
        row2.addWidget(self.spin_black)
        row2.addWidget(QLabel("白场%"))
        row2.addWidget(self.spin_white)
        self.main_layout.addLayout(row2)

        # 滑块区
        grid= QGridLayout()
        self.main_layout.addLayout(grid)
        row=0
        # 曝光
        self.slider_exp,self.spin_exp= self.create_slider_spin(-50,50,0,10)
        grid.addWidget(QLabel("曝光(-5~+5EV)"), row,0)
        grid.addWidget(self.slider_exp,row,1)
        grid.addWidget(self.spin_exp,row,2)
        row+=1
        # gamma(0.1~5)
        self.slider_gamma,self.spin_gamma= self.create_slider_spin(10,500,100,100)
        grid.addWidget(QLabel("Gamma(0.1~5.0)"), row,0)
        grid.addWidget(self.slider_gamma,row,1)
        grid.addWidget(self.spin_gamma,row,2)
        row+=1
        # 色温
        self.slider_temp,self.spin_temp= self.create_slider_spin(-50,50,0,10)
        grid.addWidget(QLabel("色温(-50~+50)"), row,0)
        grid.addWidget(self.slider_temp,row,1)
        grid.addWidget(self.spin_temp,row,2)
        row+=1
        # 色调
        self.slider_tint,self.spin_tint= self.create_slider_spin(-50,50,0,10)
        grid.addWidget(QLabel("色调(-50~+50)"), row,0)
        grid.addWidget(self.slider_tint,row,1)
        grid.addWidget(self.spin_tint,row,2)
        row+=1
        # 曲线
        self.slider_curve,self.spin_curve= self.create_slider_spin(-50,50,0,10)
        grid.addWidget(QLabel("曲线(-50~+50)"),row,0)
        grid.addWidget(self.slider_curve,row,1)
        grid.addWidget(self.spin_curve,row,2)
        row+=1
        # 饱和
        self.slider_sat,self.spin_sat= self.create_slider_spin(-50,50,0,10)
        grid.addWidget(QLabel("饱和(-50~+50)"), row,0)
        grid.addWidget(self.slider_sat,row,1)
        grid.addWidget(self.spin_sat,row,2)
        row+=1

        # 预览
        self.crop_label= CropAndRotateLabel()
        self.main_layout.addWidget(self.crop_label)

        # 数据
        self.fullres_current=None
        self.preview_base=None
        self.manual_wb=None
        self.select_wb_mode=False
        self.use_selection=False

        self.timer_sliders= QTimer(self)
        self.timer_sliders.setInterval(150)
        self.timer_sliders.setSingleShot(True)
        self.timer_sliders.timeout.connect(self.update_preview)

        self.shortcut_undo= QShortcut(QKeySequence("Ctrl+Z"), self)
        self.shortcut_undo.activated.connect(self.on_undo)
        self.undo_stack=[]

        self.crop_label.installEventFilter(self)

    def create_slider_spin(self,min_val,max_val,def_val,scale):
        """
        同步Slider & DoubleSpinBox
        """
        slider= QSlider(Qt.Horizontal)
        slider.setRange(min_val,max_val)
        slider.setValue(def_val)
        spin= QDoubleSpinBox()
        spin.setRange(min_val/scale, max_val/scale)
        spin.setSingleStep(1.0/scale)
        spin.setValue(def_val/scale)
        slider.valueChanged.connect(lambda val: spin.setValue(val/scale))
        spin.valueChanged.connect(lambda val: slider.setValue(int(round(val*scale))))
        slider.valueChanged.connect(self.on_slider_changed)
        return slider,spin

    def on_slider_changed(self):
        self.timer_sliders.start()

    def on_open(self):
        dlg= QFileDialog()
        dlg.setNameFilters(["RAW/TIFF files(*.arw *.nef *.nrw *.dng *.cr2 *.tif *.tiff)","All Files(*)"])
        if dlg.exec_():
            path= dlg.selectedFiles()[0]
            try:
                ext= os.path.splitext(path)[1].lower()
                if ext in [".arw",".nef",".nrw",".dng",".cr2"]:
                    rgb16= read_raw_widegamut(path)
                elif ext in [".tif",".tiff"]:
                    rgb16= read_tiff_anydepth(path)
                else:
                    raise ValueError("不支持文件类型:"+ ext)
                rgb_lin= rgb16.astype(np.float32)/65535.0
                neg= invert_negative(rgb_lin)
                cut= 0.001 if self.checkbox_crop.isChecked() else 0.0
                base= channel_balance_auto_levels(neg, cut, cut)
                self.fullres_current= np.clip(base,0,1)
                self.preview_base= downsample_float(self.fullres_current, 1000)
                self.manual_wb=None
                self.select_wb_mode=False
                self.use_selection=False
                self.crop_label.clearCropRect()
                self.undo_stack.clear()
                # 重置滑块
                self.slider_temp.setValue(0)
                self.slider_tint.setValue(0)
                self.slider_exp.setValue(0)
                self.slider_gamma.setValue(100)
                self.slider_curve.setValue(0)
                self.slider_sat.setValue(0)

                self.update_preview()
            except Exception as e:
                traceback.print_exc()

    def update_preview(self):
        if self.preview_base is None:
            return
        cut= 0.001 if self.checkbox_crop.isChecked() else 0.0

        # Step1: auto-level
        if self.use_selection and self.crop_label.getSelectedRect():
            x,y,w,h= self.crop_label.getSelectedRect()
            Hp, Wp, _= self.preview_base.shape
            X=int(round(x))
            Y=int(round(y))
            sel_w= int(round(w))
            sel_h= int(round(h))
            if X+sel_w>Wp: sel_w= Wp-X
            if Y+sel_h>Hp: sel_h= Hp-Y
            region_prev= self.preview_base[Y:Y+sel_h, X:X+sel_w,:]
            params= compute_autolevel_params(region_prev,0.001,0.001)
            base_after_level= apply_autolevels_with_params(self.preview_base, params)
        elif self.checkbox_crop.isChecked():
            cropped_prev= crop_image_center(self.preview_base, self.spin_black.value(), self.spin_white.value())
            params= compute_autolevel_params(cropped_prev,0.001,0.001)
            base_after_level= apply_autolevels_with_params(self.preview_base, params)
        else:
            base_after_level= channel_balance_auto_levels(self.preview_base,0.001,0.001)

        # Step2: sliders
        temp= self.slider_temp.value()
        tint= self.slider_tint.value()
        exp_val= self.slider_exp.value()/10.0
        gamma_val= self.slider_gamma.value()/100.0
        curve_val= self.slider_curve.value()
        sat_val= self.slider_sat.value()

        final= pipeline_process(base_after_level, temp,tint, exp_val,gamma_val, curve_val, sat_val, self.manual_wb)

        # Step3: 转 sRGB
        final_srgb= convert_to_srgb(final)
        preview= downsample_float(final_srgb, 1000)
        rgb_8= (np.clip(preview*255,0,255)).astype(np.uint8)
        hh,ww,_= rgb_8.shape
        qimg= QImage(rgb_8.data, ww,hh, ww*3, QImage.Format_RGB888)
        pm= QPixmap.fromImage(qimg)
        self.crop_label.setPixmap(pm)
        self.crop_label.update()

    def on_confirm_selection(self):
        if self.crop_label.getSelectedRect():
            self.use_selection=True
            print("已确认选区:仅用选区亮度范围计算autolevel参数")
        else:
            print("无选区")
            self.use_selection=False
        self.update_preview()

    def on_clear_selection(self):
        self.use_selection=False
        self.crop_label.clearCropRect()
        print("已取消选区")
        self.update_preview()

    def on_select_wb(self):
        self.select_wb_mode=True
        print("请在预览图中点选中性灰点")

    def on_clear_wb(self):
        self.manual_wb=None
        print("已清除WB")
        self.update_preview()

    def on_save(self):
        """
        保存: 用16位写出(使用 tifffile)，若能则先转ProPhoto，否则直接写uint16
        """
        if self.fullres_current is None:
            return
        dlg= QFileDialog()
        dlg.setNameFilter("TIFF files(*.tif *.tiff)")
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        if dlg.exec_():
            sp= dlg.selectedFiles()[0]
            if sp:
                out_full= self.get_pipeline_fullres()
                out_arr= float_to_uint16(out_full)  # 先把 [0,1] 转到 [0,65535]
                # 可选尝试ProPhoto
                try:
                    pil_16= Image.fromarray(out_arr, mode="RGB")
                    prophoto_pil= convert_to_prophoto_pil(pil_16)
                    out_arr= np.array(prophoto_pil)
                except Exception as e:
                    print("ProPhoto转换失败，直接保存16位uint:", e)
                # 用 tifffile写 16bit
                tifffile.imwrite(sp, out_arr)
                print("已保存:", sp)

    def on_rotate(self):
        if self.fullres_current is not None:
            self.push_undo()
            self.fullres_current= np.rot90(self.fullres_current,k=3)
            self.preview_base= downsample_float(self.fullres_current,1000)
            self.update_preview()

    def on_flip_vertical(self):
        if self.fullres_current is not None:
            self.push_undo()
            self.fullres_current= np.flipud(self.fullres_current)
            self.preview_base= downsample_float(self.fullres_current,1000)
            self.update_preview()

    def on_reset(self):
        self.slider_exp.setValue(0)
        self.slider_gamma.setValue(100)
        self.slider_temp.setValue(0)
        self.slider_tint.setValue(0)
        self.slider_curve.setValue(0)
        self.slider_sat.setValue(0)
        self.checkbox_crop.setChecked(False)
        self.spin_black.setValue(0.005)
        self.spin_white.setValue(0.005)
        self.manual_wb=None
        self.use_selection=False
        self.crop_label.clearCropRect()
        print("已重置")
        self.update_preview()

    def on_undo(self):
        if self.undo_stack:
            st= self.undo_stack.pop()
            self.fullres_current= st["fullres"].copy()
            self.slider_temp.setValue(st["temp"])
            self.slider_tint.setValue(st["tint"])
            self.slider_exp.setValue(st["exp"])
            self.slider_gamma.setValue(st["gamma"])
            self.slider_curve.setValue(st["curve"])
            self.slider_sat.setValue(st["sat"])
            self.manual_wb=None
            self.update_preview()
            print("已撤销")
        else:
            print("无可撤销操作")

    def push_undo(self):
        if self.fullres_current is not None:
            self.undo_stack.clear()
            st={
                "fullres": self.fullres_current.copy(),
                "temp":   self.slider_temp.value(),
                "tint":   self.slider_tint.value(),
                "exp":    self.slider_exp.value(),
                "gamma":  self.slider_gamma.value(),
                "curve":  self.slider_curve.value(),
                "sat":    self.slider_sat.value(),
            }
            self.undo_stack.append(st)

    def get_pipeline_fullres(self):
        """
        保存时对 fullres_current 也做 选区/黑白场 => auto-level => pipeline
        """
        if self.fullres_current is None:
            return None

        cut= 0.001 if self.checkbox_crop.isChecked() else 0.0
        if self.use_selection and self.crop_label.getSelectedRect():
            x,y,w,h= self.crop_label.getSelectedRect()
            Hp,Wp,_= self.preview_base.shape
            Hf,Wf,_= self.fullres_current.shape
            scale_x= Wf/ Wp
            scale_y= Hf/ Hp
            X= int(round(x*scale_x))
            Y= int(round(y*scale_y))
            sel_w= int(round(w*scale_x))
            sel_h= int(round(h*scale_y))
            if X+sel_w>Wf: sel_w= Wf- X
            if Y+sel_h>Hf: sel_h= Hf- Y
            region_f= self.fullres_current[Y:Y+sel_h, X:X+sel_w, :]
            params= compute_autolevel_params(region_f,0.001,0.001)
            base_after= apply_autolevels_with_params(self.fullres_current, params)
        elif self.checkbox_crop.isChecked():
            cropped_f= crop_image_center(self.fullres_current, self.spin_black.value(), self.spin_white.value())
            params= compute_autolevel_params(cropped_f,0.001,0.001)
            base_after= apply_autolevels_with_params(self.fullres_current, params)
        else:
            base_after= channel_balance_auto_levels(self.fullres_current,0.001,0.001)

        temp= self.slider_temp.value()
        tint= self.slider_tint.value()
        exp_val= self.slider_exp.value()/10.0
        gamma_val= self.slider_gamma.value()/100.0
        curve_val= self.slider_curve.value()
        sat_val= self.slider_sat.value()

        out= pipeline_process(base_after, temp,tint, exp_val, gamma_val, curve_val, sat_val,self.manual_wb)
        return out

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_preview()

def main():
    app= QApplication(sys.argv)
    win= MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__=="__main__":
    main()