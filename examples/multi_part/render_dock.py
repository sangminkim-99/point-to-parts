"""Native Viser dock for Gaussian diagnostics and per-part camera views."""
import numpy as np


class RenderDock:
    def __init__(self, server):
        self.gui = server.gui
        self.panel = self.gui.add_panel()
        self.controls = self.panel.add_tab('Gaussian diagnostics')
        self.parts = self.panel.add_tab('Part renders')
        self.panel.dock_left()
        self.panel.set_width(380)
        self.images = {}
        with self.parts:
            self.message = self.gui.add_markdown('Enable Render diagnostics. No render yet.')
            self.gui.add_markdown('Per-part Gaussian contributions in the camera frame. Held poses are not new observations.')

    def publish(self, frame, images, status):
        # Overview/observed/residual images already have a selector in controls.
        part_images = [(title, rgb) for title, rgb in images if title.startswith('Part ')]
        keys = {title.split(' — ')[0] for title, _ in part_images}
        for key in list(self.images):
            if key not in keys:
                self.images.pop(key).remove()
        with self.parts:
            for title, rgb in part_images:
                if rgb is None:
                    continue
                key = title.split(' — ')[0]
                if key not in self.images:
                    self.images[key] = self.gui.add_image(np.asarray(rgb), label=title)
                else:
                    self.images[key].image = np.asarray(rgb)
                    self.images[key].label = title
        self.message.content = f'Render frame **{frame}** · {status}'

    def status(self, message):
        self.message.content = message

    def close(self):
        self.panel.remove()
