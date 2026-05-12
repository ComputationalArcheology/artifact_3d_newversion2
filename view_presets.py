from pyvistaqt import QtInteractor

def set_view(plotter: QtInteractor, which: str, *, ortho=True):
    w = (which or "").lower()
    if w == "front":
        plotter.view_xz()
    elif w == "side":
        plotter.view_yz()
    elif w == "top":
        plotter.view_xy()
    else:
        plotter.view_isometric()
    plotter.camera.parallel_projection = bool(ortho)
    plotter.render()
