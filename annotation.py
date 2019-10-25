import numpy as np
from argparse import Namespace
import time
import json

import PyQt5 as qt
import pyqtgraph as pg

class ROIAnnotationResult(Namespace):
    """
    simple wrapper for annotation results
    """
    def __init__(self, roi, label, verified=False):
        super().__init__(
            roi = np.array(roi),
            label = int(label),
            verified = bool(verified)
        )
    def to_json(self):
        d = self.__dict__.copy()
        d['roi'] = list((int(di) for di in d['roi']))
        return d
        
        
def manually_correct_rois(img, rois, labels, colors=None):
    """
    Manually correct image (rectangle) ROI annotations
    using an interactive plot (pyqtgraph)
    
    Parameters
    ----------
    img: np-array
        the image (TODO: test with something non-grayscale?)
    rois: list of 4-element-iterable
        list of (x0,y0,w,h)-rois
    labels: list of ints
        labels, size needs to be >= size(rois)
        if a label/class does not occur in the given rois, but you wish to use
        it for manual annotation, append the label at the end of the list
        (using this, you can do de-novo annotation)
    colors: map label -> something accepted by QColor constructor
        custom colors for the labels, optional
        
    Returns
    -------
    roi_results: list of ROIAnnotationResult
        the manually curated ROIs (roi, class & whether it was manually verified or not)
    
    """
    
    # set to default image axis order
    # TODO: do we need this?
    pg.setConfigOptions(imageAxisOrder='row-major')
    
    # keep all unique labels in list
    labels_list = list(set(labels))
    
    # if no custom colors are specified
    # generate some uniformly spaced colors in hsv    
    if colors == None:
        # number of unique labels
        n_colors = len(set(labels))
        hs = np.linspace(0,255,n_colors+1)
        colors = {k : qt.QtGui.QColor.fromHsv(hs[i], 255, 255, 255) for i,k in enumerate(set(labels))}
    
    # generate QColors with a less saturated version for unverified boxes
    qt_colors = {}
    for lab in set(labels):
        c_verified = qt.QtGui.QColor(colors[lab])
        c_unverified = qt.QtGui.QColor()
        c_unverified.setHsv(*(v // 1.5 if i in (1, 2) else v for i,v in enumerate(c_verified.getHsv())))
        qt_colors[lab] = (c_verified, c_unverified)
    
    # init results
    roi_results = [ROIAnnotationResult(roi, lab, 0) for (roi, lab) in zip(rois, labels)]
    
    # init window
    win = pg.GraphicsWindow()
    win.setWindowTitle('A Test')

    # wait for window closing
    done = False
    def done_function():
        nonlocal done
        done = True
        
    # delete on close, so we can catch the destroyed signal
    win.setAttribute(qt.QtCore.Qt.WA_DeleteOnClose)
    win.destroyed.connect(done_function)
    
    # add a label
    # TODO: add a meaningful label
    label = pg.LabelItem(justify='right')
    label.setText("<span style:'color: white'>aaa</span>")
    win.addItem(label)
    
    # show the image
    ii = pg.ImageItem(img)
    v = win.addViewBox(lockAspect=True)
    v.invertY(True)
    v.addItem(ii)
    
    # keep track of mouse position in the background
    # that way, we can place new ROIs where user clicked
    last_mouse_pos = None
    def mouseMoved(evt):
        nonlocal last_mouse_pos
        last_mouse_pos=v.mapSceneToView(evt[0])
    proxy = pg.SignalProxy(ii.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)
    
    
    def connect_roi_annotation_result(new_roi_results):
        """
        make a UI-ROI for given ROIAnnotationResult
        """
        nonlocal roi_results
        
        new_roi_qt = pg.RectROI(new_roi_results.roi[:2], new_roi_results.roi[2:], removable=True)
        new_roi_qt.pen.setColor(qt_colors[new_roi_results.label][0 if new_roi_results.verified else 1])
        new_roi_qt.pen.setWidth(3)
        new_roi_qt.setAcceptedMouseButtons(qt.QtCore.Qt.LeftButton)
        
        # update coordinates upon change in UI
        def changed():
            new_roi_results.roi = list(new_roi_qt.pos()) + list(new_roi_qt.size())
        new_roi_qt.sigRegionChanged.connect(changed)
        
        # change class (and color) upon clicking in UI
        def clicked():
            if not new_roi_results.verified:
                new_roi_results.verified = True
            else:
                new_roi_results.label=labels_list[(labels_list.index(new_roi_results.label) + 1) % len(labels_list)]
            new_roi_qt.pen.setColor(qt_colors[new_roi_results.label][0 if new_roi_results.verified else 1])
        new_roi_qt.sigClicked.connect(clicked)
        
        # remove from both UI and model if requested
        def remove():
            v.removeItem(new_roi_qt)
            roi_results.remove(new_roi_results)
        new_roi_qt.sigRemoveRequested.connect(remove)
        
        # add to UI
        v.addItem(new_roi_qt)
    
    # connect all given rois and add to ui
    for roi_result in roi_results:
        connect_roi_annotation_result(roi_result)
    
    # add menu item to add new roi
    ac = v.menu.addAction('Add ROI')
    def add_roi(e):
        nonlocal last_mouse_pos
        nonlocal roi_results
        
        # create new roi at last mouse position
        new_roi = [int(last_mouse_pos.x()), int(last_mouse_pos.y()), 20,20]
        new_roi_results = ROIAnnotationResult(new_roi, labels_list[0], True)
        roi_results.append(new_roi_results)
        
        # add to ui and connect events
        connect_roi_annotation_result(new_roi_results)       
        
    ac.triggered.connect(add_roi)

    # bring window to front
    win.raise_()
    win.show()
    win.activateWindow()
    
    # wait for window closing
    while not done:
        qt.QtGui.QApplication.processEvents()
    
    return roi_results