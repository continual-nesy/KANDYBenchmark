import yaml
import numpy as np
import colorsys
import re
import pyswip
import os

from PIL import Image, ImageDraw

# Class encoding a single task. It takes the global configuration, task parameters, a list of past tasks, a seeded random number generator, and optionally a logger to print debug informations.
class Task:
    def __init__(self, id, config, task_specs, patience=1000, logger=None):
        self.task_id = id  # task_specs["id"] # Task id.
        self.name = task_specs["name"]  # Task name.
        self.beta = task_specs["beta"]  # Minimum supervision probability.
        self.gamma = task_specs["gamma"]  # Maximum supervision probability.
        self.total_samples = task_specs["samples"]  # Number of samples to generate for this task.
        self.noisy_color = task_specs["noisy_color"]  # Should the color be altered with noise?
        self.noisy_size = task_specs["noisy_size"]  # Should the size be altered with noise?
        self.rot_noise = task_specs["rot_noise"]  # Should atomic shapes be randomly rotated?

        # Color noise parameters: the RGB color is converted to HSV and then 0-mean gaussian noise is injected into each coordinate. These parameters are the standard deviation of each gaussian.
        # Values should be chosen by trial and error to preserve perceptual semantics for the entire palette of colors (ie. a human can classify a noisy yellow still as "yellow").
        self.h_sigma = config["h_sigma"]
        self.s_sigma = config["s_sigma"]
        self.v_sigma = config["v_sigma"]

        self.bg_color = config["bg_color"]  # Background color for every image.

        # Size noise parameter. A uniform random value from -noise to +noise is added to size. (eg. a base size of 25 pixels with a noise of 5 can range from 20 to 30 pixels).
        self.size_noise = config["size_noise"]

        # Available base shapes. This dictionary can be extended arbitrarily, as long as each shape is defined by: type (polygon/star/ellipse), n (number of sides/points/ratio between axes), rot (rotation angle in [0,360) ).
        self.shapes = config["shapes"]
        self.shape_names = config["shape_names"]

        # Available colors. This dictionary can be extended arbitrarily, as long as noise is adjusted accordingly.
        self.colors = config["colors"]

        # Available sizes. This dictionary can be extended arbitrarily, as long as noise is adjusted accordingly.
        self.sizes = config["sizes"]

        self.canvas_size = config["canvas_size"] # Canvas size (w, h)
        self.padding = config["padding"] # Padding from the border of the canvas (pixels)

        self.compositional_operators = set(config["compositional_operators"])
        self.list_operators = {k: [k2 for k2 in v.keys()] for k, v in config["list_operators"].items()}
        self.pregrounding_list_operators = {k: [k2 for k2 in v.keys()] for k, v in config["pregrounding_list_operators"].items()}

        self.size_values = [v for x in self.sizes for v in x.values()]

        self.seed = config["seed"] ^ self.task_id # Random seed for reproducibility.
        self.rng = np.random.RandomState(self.seed)
        self.logger = logger

        self.train_split = task_specs["train_split"]
        self.val_split = task_specs["val_split"]

        self.sample_sets = {
            "positive": task_specs["positive_set"],
            "negative": task_specs["negative_set"]
        }

        self.aliases = {}

        self.patience = patience # Number of trials for rejection sampling before giving up.
        self.rejected = {"positive": {"rule": 0, "existing": 0}, "negative": {"rule": 0, "existing": 0}}

        if "positive_rule" in task_specs:
            tmp = re.sub("[\t\n\r]+", " ", task_specs["positive_rule"])
            tmp = re.sub(" +", " ", tmp)

            self.prolog_rules = [r for r in tmp.split(".") if r != " " and r != ""]
            self.prolog_bg_knowledge = config["background_knowledge"]
            self.prolog_interpreter = config["interpreter"]
            self.prolog_check = len(self.prolog_rules) > 0 and self.prolog_bg_knowledge is not None
        else:
            self.prolog_rules = []
            self.prolog_bg_knowledge = None
            self.prolog_interpreter = None
            self.prolog_check = False

        self.shape_templates = self._generate_shapes()

        self.dirty = True # Dirty bit, if False a call to self.reset() does nothing.
        self.reset()

    def _generate_shapes(self):
        templates = {}

        size = min(self.canvas_size)

        for s, v in self.shapes.items():
            bitmap = Image.new('L', self.canvas_size, 0)
            draw = ImageDraw.Draw(bitmap)
            if v["type"] == "ellipse":
                n = float(v["n"])
                bounding_box = (
                    (self.canvas_size[0] / 2 - size / 2, self.canvas_size[1] / 2 - size / (2 * n)),
                    (self.canvas_size[0] / 2 + size / 2, self.canvas_size[1] / 2 + size / (2 * n)))
                a = int(v["rot"])
                if a == 0:
                    draw.ellipse(bounding_box, fill=255, outline=None)
                else:
                    tmp_bitmap = Image.new('L', self.canvas_size, 0)
                    tmp_draw = ImageDraw.Draw(tmp_bitmap)
                    tmp_draw.ellipse(bounding_box, fill=255, outline=None)
                    tmp_bitmap = tmp_bitmap.rotate(a,
                                                   resample=Image.Resampling.BICUBIC)  # Lanczos not possible for rotations.
                    bitmap.paste(tmp_bitmap, mask=tmp_bitmap)
            elif v["type"] == "polygon":
                xy = []
                n = int(v["n"])
                t = 2 * np.pi / n
                a = int(v["rot"]) * np.pi / 180
                for i in range(n + 1):
                    xy.append(
                        (self.canvas_size[0] / 2 + size / 2 * np.cos(t * i + a),
                         self.canvas_size[1] / 2 + size / 2 * np.sin(t * i + a))
                    )
                draw.polygon(xy, fill=255, outline=None)
            elif v["type"] == "star": # For stars, odd vertexes are at distance r (size / 2), while even vertexes are at distance r/2 (size / 4).
                xy = []
                n = int(v["n"])
                t = np.pi / n
                a = int(v["rot"]) * np.pi / 180
                for i in range(2 * n + 1):
                    if i % 2 == 1:
                        xy.append(
                            (self.canvas_size[0] / 2 + size / 2 * np.cos(t * i + a),
                             self.canvas_size[1] / 2 + size / 2 * np.sin(t * i + a))
                        )
                    else:
                        xy.append(
                            (self.canvas_size[0] / 2 + size / 4 * np.cos(t * i + a),
                             self.canvas_size[1] / 2 + size / 4 * np.sin(t * i + a))
                        )
                draw.polygon(xy, fill=255, outline=None)

            templates[s] = bitmap

        return templates


    # Loads a Swi-Prolog interpreter and executes a query. Since Swi-Prolog has a singleton database, each query requires cleaning up every predicate.
    # Creating a new interpreter for every sample is very inefficient, but for the case of shuffled curricula, it is the only way to avoid problems.
    def _check_rule(self, query):
        for r in self.prolog_rules:
            self.prolog_interpreter.assertz(r)
        out = list(self.prolog_interpreter.query(query, maxresult=1))  # Returns an empty list for fail, otherwise a list with a dict for every solution, if no variable is grounded, the dict is empty (so [{}] is a satisfiable solution).

        for r in self.prolog_rules:
            head = r.split(":-")[0]
            pred = head.split("(")[0].strip()
            for arity in range(10):  # Ugly hack to abolish all possible arities...
                next(self.prolog_interpreter.query(
                    "abolish({}/{})".format(pred, arity)))  # We abolish the predicate to avoid conflicting with other tasks.
        return len(out) > 0

    # Simple logging method.
    def _log(self, message, level="debug"):
        if self.logger is not None:
            if level == "error":
                self.logger.error(message)
            elif level == "warning":
                self.logger.warning(message)
            else:
                self.logger.debug(message)

    # Compute random (uniform) size noise.
    def _inject_size_noise(self, size):
        if self.size_noise > 0:
            rnd = self.rng.uniform(-self.size_noise, self.size_noise)
        else:
            rnd = 0
        return max(1, size + rnd)

    # Compute random (uniform) rotation noise.
    def _get_rot_noise(self):
        if self.rot_noise > 0:
            rnd = self.rng.uniform(-max(360, self.rot_noise), max(360, self.rot_noise))
        else:
            rnd = 0
        return rnd


    # colorsys requires lists, PIL requires #rrggbb strings. These utilities handle conversions.

    def _rgb_to_list(self, string):
        assert re.match("^#[0-9a-f]{6}$", string, flags=re.IGNORECASE) is not None
        rgb = re.search("#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})", string, flags=re.IGNORECASE).groups()
        rgb = [int(x, 16) for x in rgb]
        return rgb

    def _list_to_rgb(self, rgb):
        assert len(rgb) == 3
        return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

    # Inject gaussian noise into HSV coordinates.
    def _inject_color_noise(self, rgb):
        hsv = colorsys.rgb_to_hsv(*self._rgb_to_list(rgb))
        h = hsv[0]  # h in [0.0, 1.0]
        s = hsv[1]  # s in [0.0, 1.0]
        v = hsv[2]  # v in [0, 255]
        if self.h_sigma > 0.0:
            rnd = self.rng.normal(0, self.h_sigma)
            h = max(0, min(1, h + rnd * 1.0))
        if self.s_sigma > 0.0:
            rnd = self.rng.normal(0, self.s_sigma)
            s = max(0, min(1, s + rnd * 1.0))
        if self.v_sigma > 0.0:
            rnd = self.rng.normal(0, self.v_sigma)
            v = max(0, min(255, int(v + rnd * 255)))

        return self._list_to_rgb(colorsys.hsv_to_rgb(h, s, v))

    # Draw a symbol on a canvas_size image.
    def _draw_base_object(self, symbol, canvas_size):
        shape = symbol["shape"]
        color_name = symbol["color"]
        size_name = symbol["size"]

        for c in self.colors:
            if color_name in c.keys():
                color = c[color_name]

        if self.noisy_color:
            color = self._inject_color_noise(color)

        for s in self.sizes:
            if size_name in s.keys():
                size = s[size_name]

        if self.noisy_size:
            size = self._inject_size_noise(size)

        if canvas_size[0] < max(self.size_values) + self.size_noise or canvas_size[1] < max(
                self.size_values) + self.size_noise:
            self._log("Canvas size too small. Defaulting to ({},{})".format(max(self.size_values) + self.size_noise,
                                                                            max(self.size_values) + self.size_noise),
                     "warning")
            canvas_size = (max(self.size_values) + self.size_noise, max(self.size_values) + self.size_noise)

        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))


        tmp = Image.new('RGBA', self.canvas_size, color=color)
        tmp.putalpha(self.shape_templates[shape].getchannel('L'))

        if self.rot_noise > 0:
            tmp = tmp.rotate(self._get_rot_noise(), resample=Image.Resampling.BICUBIC)

        tmp = tmp.resize((int(size), int(size)), resample=Image.Resampling.LANCZOS)
        bitmap.paste(tmp, (int(canvas_size[0] / 2 - size / 2), int(canvas_size[1] / 2 - size / 2)), mask=tmp)


        return bitmap

    # "in" composition function. Recursively draw each element from the symbol list in a canvas_size image.
    def _draw_in(self, symbol, canvas_size):
        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        for s in symbol["in"]:
            bmp = self._draw(s, canvas_size)
            bitmap.paste(bmp, mask=bmp)

        return bitmap

    # "quadrant_ul" composition function. Recursively draw each element from the symbol list in a canvas_size image.
    def _draw_quadrant_ul(self, symbol, canvas_size):
        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        for s in symbol["quadrant_ul"]:
            bmp = self._draw(s, (canvas_size[0] // 2, canvas_size[1] // 2))
            x0 = 0
            y0 = 0
            bitmap.paste(bmp, (x0, y0), mask=bmp)

        return bitmap

    # "quadrant_ul" composition function. Recursively draw each element from the symbol list in a canvas_size image.
    def _draw_quadrant_ur(self, symbol, canvas_size):
        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        for s in symbol["quadrant_ur"]:
            bmp = self._draw(s, (canvas_size[0] // 2, canvas_size[1] // 2))
            x0 = int(canvas_size[0] / 2)
            y0 = 0
            bitmap.paste(bmp, (x0, y0), mask=bmp)

        return bitmap

    # "quadrant_ul" composition function. Recursively draw each element from the symbol list in a canvas_size image.
    def _draw_quadrant_ll(self, symbol, canvas_size):
        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        for s in symbol["quadrant_ll"]:
            bmp = self._draw(s, (canvas_size[0] // 2, canvas_size[1] // 2))
            x0 = int(canvas_size[0] / 2)
            y0 = 0
            bitmap.paste(bmp, (x0, y0), mask=bmp)

        return bitmap

    # "quadrant_ul" composition function. Recursively draw each element from the symbol list in a canvas_size image.
    def _draw_quadrant_lr(self, symbol, canvas_size):
        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        for s in symbol["quadrant_lr"]:
            bmp = self._draw(s, (canvas_size[0] // 2, canvas_size[1] // 2))
            x0 = int(canvas_size[0] / 2)
            y0 = int(canvas_size[1] / 2)
            bitmap.paste(bmp, (x0, y0), mask=bmp)

        return bitmap

    # "stack" composition function. Recursively draw each element from the symbol list in a canvas_size image.
    def _draw_stack(self, symbol, canvas_size, reduce_bounding_box=False):
        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        if reduce_bounding_box:
            child_canvas = (self.size_noise + max(*self.size_values, canvas_size[0] // len(symbol["stack_reduce_bb"])),
                            self.size_noise + max(*self.size_values, canvas_size[1] // len(symbol["stack_reduce_bb"])))
        else:
            child_canvas = (self.size_noise + max(*self.size_values, canvas_size[0]),
                            self.size_noise + max(*self.size_values, canvas_size[1] // len(symbol["stack"])))
        bitmaps = []

        if reduce_bounding_box:
            for s in symbol["stack_reduce_bb"]:
                bmp = self._draw(s, child_canvas)
                bitmaps.append(bmp)
        else:
            for s in symbol["stack"]:
                bmp = self._draw(s, child_canvas)
                bitmaps.append(bmp)

        step = canvas_size[1] / (len(bitmaps) + 1)

        for i in range(len(bitmaps)):
            if reduce_bounding_box:
                x0 = int(canvas_size[0] / 2 - child_canvas[0] / 2)
            else:
                x0 = int(0)
            y0 = int((i + 1) * step - child_canvas[1] / 2)

            bitmap.paste(bitmaps[i], (x0, y0), mask=bitmaps[i])

        return bitmap

    # "side_by_side" composition function. Recursively draw each element from the symbol list in a canvas_size image.
    def _draw_sbs(self, symbol, canvas_size, reduce_bounding_box=False):
        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        if reduce_bounding_box:
            child_canvas = (self.size_noise + max(*self.size_values, canvas_size[0] // len(symbol["side_by_side_reduce_bb"])),
                            self.size_noise + max(*self.size_values, canvas_size[1] // len(symbol["side_by_side_reduce_bb"])))
        else:
            child_canvas = (self.size_noise + max(*self.size_values, canvas_size[0] // len(symbol["side_by_side"])),
                            self.size_noise + max(*self.size_values, canvas_size[1]))


        bitmaps = []

        if reduce_bounding_box:
            for s in symbol["side_by_side_reduce_bb"]:
                bmp = self._draw(s, child_canvas)
                bitmaps.append(bmp)
        else:
            for s in symbol["side_by_side"]:
                bmp = self._draw(s, child_canvas)
                bitmaps.append(bmp)

        step = canvas_size[0] / (len(bitmaps) + 1)

        for i in range(len(bitmaps)):
            if reduce_bounding_box:
                y0 = int(canvas_size[1] / 2 - child_canvas[1] / 2)
            else:
                y0 = int(0)
            x0 = int((i + 1) * step - child_canvas[0] / 2)

            bitmap.paste(bitmaps[i], (x0, y0), mask=bitmaps[i])

        return bitmap

    # "diag_ul_lr" composition function. Recursively draw each element from the symbol list in a canvas_size image.
    def _draw_ullr(self, symbol, canvas_size):
        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        child_canvas = (self.size_noise + max(*self.size_values, canvas_size[0] // len(symbol["diag_ul_lr"])),
                        self.size_noise + max(*self.size_values, canvas_size[1] // len(symbol["diag_ul_lr"])))

        bitmaps = []

        for s in symbol["diag_ul_lr"]:
            bmp = self._draw(s, child_canvas)
            bitmaps.append(bmp)

        step = 1 / (len(bitmaps) + 1)

        for i in range(len(bitmaps)):
            d = (i + 1) * step
            x0 = int(d * canvas_size[0] - child_canvas[0] / 2)
            y0 = int(d * canvas_size[1] - child_canvas[1] / 2)

            bitmap.paste(bitmaps[i], (x0, y0), mask=bitmaps[i])

        return bitmap

    # "diag_ll_ur" composition function. Recursively draw each element from the symbol list in a canvas_size image.
    def _draw_llur(self, symbol, canvas_size):
        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        child_canvas = (self.size_noise + max(*self.size_values, canvas_size[0] // len(symbol["diag_ll_ur"])),
                        self.size_noise + max(*self.size_values, canvas_size[1] // len(symbol["diag_ll_ur"])))

        bitmaps = []

        for s in symbol["diag_ll_ur"]:
            bmp = self._draw(s, child_canvas)
            bitmaps.append(bmp)

        step = 1 / (len(bitmaps) + 1)

        for i in range(len(bitmaps)):
            d = (len(bitmaps) - i) * step

            x0 = int(canvas_size[0] - d * canvas_size[0] - child_canvas[0] / 2)
            y0 = int(d * canvas_size[1] - child_canvas[1] / 2)

            bitmap.paste(bitmaps[i], (x0, y0), mask=bitmaps[i])

        return bitmap

    # "grid" composition function. Recursively draw each element from the symbol list in a canvas_size image.
    def _draw_grid(self, symbol, canvas_size):
        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        n = int(np.ceil(np.sqrt(len(symbol["grid"]))))

        child_canvas = (self.size_noise + max(*self.size_values, canvas_size[0] // n),
                        self.size_noise + max(*self.size_values, canvas_size[1] // n))

        bitmaps = []

        for s in symbol["grid"]:
            bmp = self._draw(s, child_canvas)
            bitmaps.append(bmp)

        for i in range(len(bitmaps)):
            j = (i % n)
            k = i // n

            x0 = int(j * canvas_size[0] / n)
            y0 = int(k * canvas_size[1] / n)

            bitmap.paste(bitmaps[i], (x0, y0), mask=bitmaps[i])

        return bitmap

    # "random" composition function. Recursively draw each element from the symbol list in a canvas_size image.
    def _draw_random(self, symbol, canvas_size):
        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        for s in symbol["random"]:
            bmp = self._draw(s, canvas_size)
            for i in range(self.patience): # Greedy rejection sampling. Tries to fit the next element, with no backtracking.
                m = max(self.size_values) + self.size_noise
                x0 = self.rng.randint(m // 2 + self.padding // 2, canvas_size[0] - m // 2 - self.padding // 2) - canvas_size[0] // 2
                y0 = self.rng.randint(m // 2 + self.padding // 2, canvas_size[1] - m // 2 - self.padding // 2) - canvas_size[1] // 2
                x1 = x0 + canvas_size[0]
                y1 = y0 + canvas_size[1]

                tmp_bmp = Image.new('RGBA', canvas_size, (0,0,0,0))
                tmp_bmp.paste(bmp, (x0, y0, x1, y1), mask=bmp)

                global_mask = np.array(bitmap.split()[-1], dtype=bool)
                new_mask = np.array(tmp_bmp.split()[-1], dtype=bool)

                if np.sum(np.logical_and(global_mask, new_mask)) == 0:
                    break
            bitmap.paste(bmp, (x0, y0, x1, y1), mask=bmp)

        if i == self.patience:
            self._log("Ran out of patience while positioning {} random objects. There will be overlap.".format(symbol["random"]), "warning")

        return bitmap

    # Wrapper for the recursive drawing procedure.
    def _draw(self, symbol, canvas_size):
        if "in" in symbol.keys():
            return self._draw_in(symbol, canvas_size)
        elif "stack" in symbol.keys():
            return self._draw_stack(symbol, canvas_size)
        elif "stack_reduce_bb" in symbol.keys():
            return self._draw_stack(symbol, canvas_size, reduce_bounding_box=True)
        elif "side_by_side" in symbol.keys():
            return self._draw_sbs(symbol, canvas_size)
        elif "side_by_side_reduce_bb" in symbol.keys():
            return self._draw_sbs(symbol, canvas_size, reduce_bounding_box=True)
        elif "diag_ul_lr" in symbol.keys():
            return self._draw_ullr(symbol, canvas_size)
        elif "diag_ll_ur" in symbol.keys():
            return self._draw_llur(symbol, canvas_size)
        elif "quadrant_ul" in symbol.keys():
            return self._draw_quadrant_ul(symbol, canvas_size)
        elif "quadrant_ur" in symbol.keys():
            return self._draw_quadrant_ur(symbol, canvas_size)
        elif "quadrant_ll" in symbol.keys():
            return self._draw_quadrant_ll(symbol, canvas_size)
        elif "quadrant_lr" in symbol.keys():
            return self._draw_quadrant_lr(symbol, canvas_size)
        elif "grid" in symbol.keys():
            return self._draw_grid(symbol, canvas_size)
        elif "random" in symbol.keys():
            return self._draw_random(symbol, canvas_size)
        else:
            return self._draw_base_object(symbol, canvas_size)

    # Upper level wrapper. Draws the sample and then removes transparency and adds a padding.
    def _draw_sample(self, symbol):
        w = self.canvas_size[0] - self.padding
        h = self.canvas_size[1] - self.padding

        bitmap = self._draw(symbol, (w, h))

        out = Image.new('RGBA', self.canvas_size, self.bg_color)
        out.paste(bitmap, (self.padding // 2, self.padding // 2), mask=bitmap)

        return out.convert('RGB')

    def _symbol_to_prolog(self, symbol):
        if "shape" in symbol:
            return "{}_{}_{}".format(symbol["shape"], symbol["color"], symbol["size"])
        else:
            op = list(symbol.keys())[0]
            return "{}([{}])".format(op, ", ".join([self._symbol_to_prolog(x) for x in symbol[op]]))

    # Upper level wrapper. Samples a random symbol from a sample set (positive/negative).
    def _sample_symbol(self, sample_set):
        assert sample_set in ["positive", "negative"]

        self.dirty = True
        sample = self._recursive_sampling(self._pre_expand_symbol(self.rng.choice(self.sample_sets[sample_set])))

        return sample

    # Wrapper for the operator pre-expansion procedure. It descends on the structure and replaces list operators with expanded lists, flattening singleton entries.
    # It is performed BEFORE grounding to allow variability (in contrast with _expand_symbol() which is performed afterwards).
    def _pre_expand_symbol(self, symbol):
        if isinstance(symbol, dict):
            prelist_ops = list(set(symbol.keys()).intersection(set(self.pregrounding_list_operators.keys())))
            list_ops = list(set(symbol.keys()).intersection(set(self.list_operators.keys())))
            comp_ops = list(set(symbol.keys()).intersection(self.compositional_operators))

            if len(prelist_ops) + len(list_ops) + len(comp_ops) == 0:
                out = symbol
            elif len(comp_ops) == 1:
                out = {comp_ops[0]: self._pre_expand_symbol(symbol[comp_ops[0]])}
            elif len(list_ops) == 1:
                op = list_ops[0]
                if len(self.list_operators[op]) == 0:
                    out = {op: self._pre_expand_symbol(symbol[op])}
                else:
                    out = {op: {}}
                    for k in self.list_operators[op]:
                        out[op][k] = symbol[op][k]
                    if op != "recall":
                        out[op]["list"] = self._pre_expand_symbol(symbol[op]["list"])
            elif len(prelist_ops) == 1:
                op = prelist_ops[0]
                if len(self.pregrounding_list_operators[op]) == 0:
                    if op == "permute_before":
                        out = self._expand_permute(self._pre_expand_symbol(symbol[op]))
                    elif op == "palindrome_before":
                        out = self._expand_palindrome(self._pre_expand_symbol(symbol[op]))
                    elif op == "mirror_before":
                        out = self._expand_mirror(self._pre_expand_symbol(symbol[op]))
                    elif op == "random_shift_before":
                        out = self._expand_random_shift(self._pre_expand_symbol(symbol[op]))
                    elif op == "any_composition":
                        new_op = self.rng.choice(list(self.compositional_operators))
                        out = {new_op: self._pre_expand_symbol(symbol[op])}
                    elif op == "any_displacement":
                        new_op = self.rng.choice(list(set(self.compositional_operators) - {"random", "in", "quadrant_ul", "quadrant_ur", "quadrant_ll", "quadrant_lr"}))
                        out = {new_op: self._pre_expand_symbol(symbol[op])}
                    elif op == "any_line":
                        new_op = self.rng.choice(list(set(self.compositional_operators) - {"random", "in", "grid", "quadrant_ul", "quadrant_ur", "quadrant_ll", "quadrant_lr"}))
                        out = {new_op: self._pre_expand_symbol(symbol[op])}
                    elif op == "any_diag":
                        new_op = self.rng.choice(["diag_ul_lr", "diag_ll_ur"])
                        out = {new_op: self._pre_expand_symbol(symbol[op])}
                    elif op == "any_non_diag":
                        new_op = self.rng.choice(["stack", "side_by_side"])
                        out = {new_op: self._pre_expand_symbol(symbol[op])}
                    elif op == "any_quadrant":
                        new_op = self.rng.choice(["quadrant_ul", "quadrant_ur", "quadrant_ll", "quadrant_lr"])
                        out = {new_op: self._pre_expand_symbol(symbol[op])}
                    elif op == "quadrant_or_center":
                        new_op = self.rng.choice(["quadrant_ul", "quadrant_ur", "quadrant_ll", "quadrant_lr", "in"])
                        out = {new_op: self._pre_expand_symbol(symbol[op])}
                    elif op == "union":
                        out = self._expand_union(symbol[op])
                    elif op == "intersection":
                        out = self._expand_intersection(symbol[op])
                    elif op == "difference":
                        out = self._expand_difference(symbol[op])
                    elif op == "symmetric_difference":
                        out = self._expand_symmetric_difference(symbol[op])
                else:
                    if op == "sample_before":
                        out = self._expand_sample(self._pre_expand_symbol(symbol[op]["list"]), symbol[op]["n"])
                    elif op == "pick_before":
                        out = self._expand_pick(self._pre_expand_symbol(symbol[op]["list"]), symbol[op]["n"])
                    elif op == "first_before":
                        out = self._expand_first(self._pre_expand_symbol(symbol[op]["list"]), symbol[op]["n"])
                    elif op == "last_before":
                        out = self._expand_last(self._pre_expand_symbol(symbol[op]["list"]), symbol[op]["n"])
                    elif op == "shift_before":
                        out = self._expand_shift(self._pre_expand_symbol(symbol[op]["list"]), symbol[op]["n"])
                    elif op == "repeat_before":
                        out = self._expand_repeat(self._pre_expand_symbol(symbol[op]["list"]), symbol[op]["n"])
                    elif op == "random_repeat_before":
                        out = self._expand_random_repeat(self._pre_expand_symbol(symbol[op]["list"]), symbol[op]["min"],
                                                         symbol[op]["max"])
                    elif op == "store_before":
                        out = self._expand_store(self._pre_expand_symbol(symbol[op]["list"]), symbol[op]["alias"])
        else:
            out = []
            for s in symbol:
                tmp = self._pre_expand_symbol(s)
                if isinstance(tmp, list):
                    out.extend(tmp)
                else:
                    out.append(tmp)

        return out

    # Wrapper for the recursive sampling procedure. It descends on the structure and grounds negations (not_), disjuntions (a|b) or don't care (~) attributes.
    def _recursive_sampling(self, symbol):
        if isinstance(symbol, dict):
            sizes = [list(x.keys())[0] for x in self.sizes]
            colors = [list(x.keys())[0] for x in self.colors]

            out = {}

            comp_ops = list(set(symbol.keys()).intersection(self.compositional_operators))
            list_ops = list(set(symbol.keys()).intersection(set(self.list_operators.keys())))
            operators = comp_ops + list_ops

            if len(operators) == 0:

                if symbol["shape"] is None:
                    out["shape"] = self.rng.choice(self.shape_names)
                elif symbol["shape"].startswith("not_"):
                    out["shape"] = self.rng.choice(list(set(self.shape_names) - {symbol["shape"][4:]}))
                else:
                    out["shape"] = self.rng.choice(symbol["shape"].split("|"))

                if symbol["color"] is None:
                    out["color"] = self.rng.choice(colors)
                elif symbol["color"].startswith("not_"):
                    out["color"] = self.rng.choice(list(set(colors) - {symbol["color"][4:]}))
                else:
                    out["color"] = self.rng.choice(symbol["color"].split("|"))

                if symbol["size"] is None:
                    out["size"] = self.rng.choice(sizes)
                elif symbol["size"].startswith("not_"):
                    out["size"] = self.rng.choice(list(set(sizes) - {symbol["size"][4:]}))
                else:
                    out["size"] = self.rng.choice(symbol["size"].split("|"))
            elif len(comp_ops) == 1:
                out[comp_ops[0]] = self._recursive_sampling(symbol[comp_ops[0]])
            elif len(list_ops) == 1:
                if len(self.list_operators[list_ops[0]]) == 0:
                    out[list_ops[0]] = self._recursive_sampling(symbol[list_ops[0]])
                else:
                    if list_ops[0] == "recall":
                        out = self._recursive_sampling(self._expand_recall(symbol[list_ops[0]]["alias"]))
                    elif list_ops[0] == "store":
                        out = self._expand_store(self._recursive_sampling(symbol[list_ops[0]]["list"]), symbol[list_ops[0]]["alias"])
                    else:
                        out[list_ops[0]] = {}
                        for k in self.list_operators[list_ops[0]]:
                            out[list_ops[0]][k] = symbol[list_ops[0]][k]

                        out[list_ops[0]]["list"] = self._recursive_sampling(symbol[list_ops[0]]["list"])

        else:
            out = [self._recursive_sampling(s) for s in symbol]

        return out


    # Wrapper for the operator expansion procedure. It descends on the structure and replaces list operators with expanded lists, flattening singleton entries.
    # In order to preserve semantics of many operators (palindrome, repeat, etc.), it must be performed AFTER sampling atomic attributes.
    def _expand_symbol(self, symbol):
        if isinstance(symbol, dict):
            list_ops = list(set(symbol.keys()).intersection(set(self.list_operators.keys())))
            comp_ops = list(set(symbol.keys()).intersection(self.compositional_operators))

            if len(list_ops) + len(comp_ops) == 0:
                out = symbol
            elif len(comp_ops) == 1:
                out = {comp_ops[0]: self._expand_symbol(symbol[comp_ops[0]])}
            elif len(list_ops) == 1:
                op = list_ops[0]
                if len(self.list_operators[op]) == 0:
                    if op == "permute":
                        out = self._expand_permute(self._expand_symbol(symbol[op]))
                    elif op == "palindrome":
                        out = self._expand_palindrome(self._expand_symbol(symbol[op]))
                    elif op == "mirror":
                        out = self._expand_mirror(self._expand_symbol(symbol[op]))
                    elif op == "random_shift":
                        out = self._expand_random_shift(self._expand_symbol(symbol[op]))
                else:
                    if op == "sample":
                        out = self._expand_sample(self._expand_symbol(symbol[op]["list"]), symbol[op]["n"])
                    elif op == "pick":
                        out = self._expand_pick(self._expand_symbol(symbol[op]["list"]), symbol[op]["n"])
                    elif op == "first":
                        out = self._expand_first(self._expand_symbol(symbol[op]["list"]), symbol[op]["n"])
                    elif op == "last":
                        out = self._expand_last(self._expand_symbol(symbol[op]["list"]), symbol[op]["n"])
                    elif op == "shift":
                        out = self._expand_shift(self._expand_symbol(symbol[op]["list"]), symbol[op]["n"])
                    elif op == "repeat":
                        out = self._expand_repeat(self._expand_symbol(symbol[op]["list"]), symbol[op]["n"])
                    elif op == "sort":
                        out = self._expand_sort(self._expand_symbol(symbol[op]["list"]), symbol[op]["order"],
                                                symbol[op]["keys"])
                    elif op == "argsort":
                        out = self._expand_argsort(self._expand_symbol(symbol[op]["list"]), symbol[op]["idx"])
                    elif op == "random_repeat":
                        out = self._expand_random_repeat(self._expand_symbol(symbol[op]["list"]), symbol[op]["min"],
                                                         symbol[op]["max"])
                    elif op == "store":
                        out = self._expand_store(self._expand_symbol(symbol[op]["list"]), symbol[op]["alias"])
                    elif op == "recall":
                        out = self._expand_recall(symbol[op]["alias"])
        else:
            out = []
            for s in symbol:
                tmp = self._expand_symbol(s)
                if isinstance(tmp, list):
                    out.extend(tmp)
                else:
                    out.append(tmp)

        return out

    # Expands a palindrome() list. The final element acts as a pivot, hence it is not repeated.
    def _expand_palindrome(self, l):
        return l + l[-2::-1]

    # Expands a mirror() list. The final element is repeated as well.
    def _expand_mirror(self, l):
        return l + l[-1::-1]

    # Expands a permute() list with a random permutation.
    def _expand_permute(self, l):
        return list(self.rng.permutation(l))

    # Expands a sample() list. It returns a random sampling with replacement.
    def _expand_sample(self, l, n):
        return list(self.rng.choice(l, n, replace=True))

    # Expands a pick() list. It returns a random sampling without replacement.
    def _expand_pick(self, l, n):
        return list(self.rng.choice(l, n, replace=False))

    # Expands a first(n) list. It returns the first n elements of the list.
    def _expand_first(self, l, n):
        return l[:n]

    # Expands a last(n) list. It returns the last n elements of the list (order is not inverted).
    def _expand_last(self, l, n):
        return l[-n:]

    # Expands a shift(n) list. It returns a barrel shift of n steps (>0 to the right, <0 to the left) of the list.
    def _expand_shift(self, l, n):
        return l[n % len(l):] + l[:n % len(l)]

    # Expands a random_shift() list. It returns a barrel shift of the list by a random amount.
    def _expand_random_shift(self, l):
        n = self.rng.randint(len(l))
        return l[n % len(l):] + l[:n % len(l)]

    # Expands a repeat(n) list. It returns n copies of the list.
    def _expand_repeat(self, l, n):
        return l * n

    # Expands a sort(order, keys) list. It returns the first n elements of the list.
    # Order: asc, desc.
    # Keys: n (number of sub-objects), shape (circle < triangle < square), color (), size (small < large).
    # If the list has at least one composition operator, the only valid key is n. Atomic objects have n=1, compositional objects have n=len(children), no recursion is performed to count total leaves.
    # Multiple keys are evaluated left to right, so sorting by [shape, color] means that the key color will affect only the order between objects with the same shape.
    def _expand_sort(self, l, order, keys):
        assert re.match("^(asc|desc)$", order,
                        flags=re.IGNORECASE) is not None, "Invalid order, expected ['asc', 'desc'], found: {}".format(
            order)

        possible_keys = ["n", "shape", "color", "size"]
        for k in keys:
            assert k in possible_keys, "Invalid sorting key, expected {}, found {}.".format(possible_keys, k)

        has_compositions = False
        for x in l:
            has_compositions = has_compositions or "shape" not in x.keys()

        assert not has_compositions or len(keys) == 1 and keys[
            0] == "n", "The only allowed sorting key for lists with non-atomic elements is ['n'], found {}.".format(
            keys)

        # Assign a weight for each key:
        sorting_weights = {"n": 0, "shape": 0, "color": 0, "size": 0}
        for i, k in enumerate(keys):
            sorting_weights[k] = 10 ** (
                        2 * (4 - i))  # n-shape-color-size -> n = 100000000, shape = 1000000, color = 10000, size = 100.

        sorting_scores = []
        if not has_compositions:
            for x in l:
                score = 0
                if "shape" in x.keys():
                    score += sorting_weights["n"] * 1

                    score += sorting_weights["shape"] * (self.shape_names.index(x["shape"]) + 1)
                    for i, y in enumerate(self.colors):
                        if x["color"] in y.keys():
                            score += sorting_weights["color"] * (i + 1)
                    for i, y in enumerate(
                            self.sizes):  # Sort by the order in self.sizes list, not by effective pixel-size
                        if x["size"] in y.keys():
                            score += sorting_weights["size"] * y[x["size"]]

                if order == "desc":
                    score *= -1
                sorting_scores.append(score)

        else:
            for x in l:
                if "shape" in x.keys():
                    score = sorting_weights["n"] * 1
                else:
                    score = sorting_weights["n"] * len(x[list(x.keys())[0]])

                if order == "desc":
                    score *= -1
                sorting_scores.append(score)

        sorted_idx = np.argsort(sorting_scores, kind="stable")  # Numpy defaults to quicksort, which is not stable.

        return [l[i] for i in sorted_idx]

    # Expands an argsort(idx) list. It returns the specified permutation of the list.
    def _expand_argsort(self, l, idx):
        assert isinstance(l, list)
        assert isinstance(idx, list)
        assert len(l) == len(idx)

        return [l[i] for i in idx]

    # Expands a random_repeat(min, max) list. It returns a random number of copies of the list between min and max (inclusive).
    def _expand_random_repeat(self, l, min, max):
        assert min < max
        n = self.rng.randint(min, max + 1)
        return l * n

    # Expands a store(alias) list. It returns the list itself, but memorizes it internally for later retrieval. It works both before and after grounding.
    def _expand_store(self, l, alias):
        self.aliases[alias] = l
        return l

    # Expands a recall(alias) operator (it receives no lists). It returns the retrieved structure previously stored.
    def _expand_recall(self, alias):
        assert alias in self.aliases, "Alias {} not stored. Make sure you call store(alias, object) or store_before(alias, object) before recall(alias). If used inside set operators, recall can only retrieve store_before aliases.".format(alias)
        return self.aliases[alias]

    # Returns the set intersection of attributes in the list of objects. The list can only contain atomic objects and this function can only be called before grounding.
    def _expand_union(self, l):
        out_shapes = set()
        out_colors = set()
        out_sizes = set()

        sizes = [list(x.keys())[0] for x in self.sizes]
        colors = [list(x.keys())[0] for x in self.colors]

        for i, x in enumerate(l):
            if "recall" in x:
                x = self._expand_recall(x["recall"]["alias"])
                assert isinstance(x, dict) or isinstance(x, list) and len(x) == 1, "Recalled shapes inside union() can only be atomic."
                if isinstance(x, list):
                    x = x[0]
            elif "union" in x:
                x = self._expand_union(x["union"])
            elif "intersection" in x:
                x = self._expand_intersection(x["intersection"])
            elif "difference" in x:
                x = self._expand_difference(x["difference"])
            elif "symmetric_difference" in x:
                x = self._expand_symmetric_difference(x["symmetric_difference"])

            assert "shape" in x, "union() operator can only process atomic shapes."

            if x["shape"] is None:
                if i == 0:
                    tmp_shape = self.shape_names
                else:
                    tmp_shape = []
            elif x["shape"].startswith("not_"):
                tmp_shape = set(self.shape_names) - {x["shape"][4:]}
            else:
                tmp_shape = x["shape"].split("|")

            if x["color"] is None:
                if i == 0:
                    tmp_color = colors
                else:
                    tmp_color = []
            elif x["color"].startswith("not_"):
                tmp_color = set(colors) - {x["color"][4:]}
            else:
                tmp_color = x["color"].split("|")

            if x["size"] is None:
                if i == 0:
                    tmp_size = sizes
                else:
                    tmp_size = []
            elif x["size"].startswith("not_"):
                tmp_size = set(sizes) - {x["size"][4:]}
            else:
                tmp_size = x["size"].split("|")

            if i == 0:
                out_shapes = set(tmp_shape)
                out_colors = set(tmp_color)
                out_sizes = set(tmp_size)
            else:
                out_shapes = out_shapes.union(set(tmp_shape))
                out_colors = out_colors.union(set(tmp_color))
                out_sizes = out_sizes.union(set(tmp_size))

        assert len(out_shapes) > 0, "union() produced no valid shape."
        assert len(out_colors) > 0, "union() produced no valid size."
        assert len(out_sizes) > 0, "union() produced no valid size."

        return {"shape": "|".join(out_shapes), "color": "|".join(out_colors), "size": "|".join(out_sizes)}

    # Returns the set intersection of attributes in the list of objects. The list can only contain atomic objects and this function can only be called before grounding.
    def _expand_intersection(self, l):
        out_shapes = set()
        out_colors = set()
        out_sizes = set()

        sizes = [list(x.keys())[0] for x in self.sizes]
        colors = [list(x.keys())[0] for x in self.colors]

        for i, x in enumerate(l):
            if "recall" in x:
                x = self._expand_recall(x["recall"]["alias"])
                assert isinstance(x, dict) or isinstance(x, list) and len(x) == 1, "Recalled shapes inside intersection() can only be atomic."
                if isinstance(x, list):
                    x = x[0]
            elif "union" in x:
                x = self._expand_union(x["union"])
            elif "intersection" in x:
                x = self._expand_intersection(x["intersection"])
            elif "difference" in x:
                x = self._expand_difference(x["difference"])
            elif "symmetric_difference" in x:
                x = self._expand_symmetric_difference(x["symmetric_difference"])

            assert "shape" in x, "intersection() operator can only process atomic shapes."

            if x["shape"] is None:
                tmp_shape = self.shape_names
            elif x["shape"].startswith("not_"):
                tmp_shape = set(self.shape_names) - {x["shape"][4:]}
            else:
                tmp_shape = x["shape"].split("|")

            if x["color"] is None:
                tmp_color = colors
            elif x["color"].startswith("not_"):
                tmp_color = set(colors) - {x["color"][4:]}
            else:
                tmp_color = x["color"].split("|")

            if x["size"] is None:
                tmp_size = sizes
            elif x["size"].startswith("not_"):
                tmp_size = set(sizes) - {x["size"][4:]}
            else:
                tmp_size = x["size"].split("|")

            if i == 0:
                out_shapes = set(tmp_shape)
                out_colors = set(tmp_color)
                out_sizes = set(tmp_size)
            else:
                out_shapes = out_shapes.intersection(set(tmp_shape))
                out_colors = out_colors.intersection(set(tmp_color))
                out_sizes = out_sizes.intersection(set(tmp_size))

        assert len(out_shapes) > 0, "intersection() produced no valid shape."
        assert len(out_colors) > 0, "intersection() produced no valid size."
        assert len(out_sizes) > 0, "intersection() produced no valid size."

        return {"shape": "|".join(out_shapes), "color": "|".join(out_colors), "size": "|".join(out_sizes)}

    # Returns the set difference of attributes in the list of objects. The list can only contain atomic objects and this function can only be called before grounding.
    def _expand_difference(self, l):
        out_shapes = set()
        out_colors = set()
        out_sizes = set()

        sizes = [list(x.keys())[0] for x in self.sizes]
        colors = [list(x.keys())[0] for x in self.colors]

        for i, x in enumerate(l):
            if "recall" in x:
                x = self._expand_recall(x["recall"]["alias"])
                assert isinstance(x, dict) or isinstance(x, list) and len(x) == 1, "Recalled shapes inside difference() can only be atomic."
                if isinstance(x, list):
                    x = x[0]
            elif "union" in x:
                x = self._expand_union(x["union"])
            elif "intersection" in x:
                x = self._expand_intersection(x["intersection"])
            elif "difference" in x:
                x = self._expand_difference(x["difference"])
            elif "symmetric_difference" in x:
                x = self._expand_symmetric_difference(x["symmetric_difference"])

            assert "shape" in x, "difference() operator can only process atomic shapes."

            if x["shape"] is None:
                if i == 0:
                    tmp_shape = self.shape_names
                else:
                    tmp_shape = []
            elif x["shape"].startswith("not_"):
                tmp_shape = set(self.shape_names) - {x["shape"][4:]}
            else:
                tmp_shape = x["shape"].split("|")

            if x["color"] is None:
                if i == 0:
                    tmp_color = colors
                else:
                    tmp_color = []
            elif x["color"].startswith("not_"):
                tmp_color = set(colors) - {x["color"][4:]}
            else:
                tmp_color = x["color"].split("|")

            if x["size"] is None:
                if i == 0:
                    tmp_size = sizes
                else:
                    tmp_size = []
            elif x["size"].startswith("not_"):
                tmp_size = set(sizes) - {x["size"][4:]}
            else:
                tmp_size = x["size"].split("|")

            if i == 0:
                out_shapes = set(tmp_shape)
                out_colors = set(tmp_color)
                out_sizes = set(tmp_size)
            else:
                out_shapes = out_shapes.difference(set(tmp_shape))
                out_colors = out_colors.difference(set(tmp_color))
                out_sizes = out_sizes.difference(set(tmp_size))

        assert len(out_shapes) > 0, "difference() produced no valid shape."
        assert len(out_colors) > 0, "difference() produced no valid size."
        assert len(out_sizes) > 0, "difference() produced no valid size."

        return {"shape": "|".join(out_shapes), "color": "|".join(out_colors), "size": "|".join(out_sizes)}

    # Returns the set symmetric difference of attributes in the list of objects. The list can only contain atomic objects and this function can only be called before grounding.
    def _expand_symmetric_difference(self, l):
        out_shapes = set()
        out_colors = set()
        out_sizes = set()

        sizes = [list(x.keys())[0] for x in self.sizes]
        colors = [list(x.keys())[0] for x in self.colors]

        for i, x in enumerate(l):
            if "recall" in x:
                x = self._expand_recall(x["recall"]["alias"])
                assert isinstance(x, dict) or isinstance(x, list) and len(x) == 1, "Recalled shapes inside symmetric_difference() can only be atomic."
                if isinstance(x, list):
                    x = x[0]
            elif "union" in x:
                x = self._expand_union(x["union"])
            elif "intersection" in x:
                x = self._expand_intersection(x["intersection"])
            elif "difference" in x:
                x = self._expand_difference(x["difference"])
            elif "symmetric_difference" in x:
                x = self._expand_symmetric_difference(x["symmetric_difference"])

            assert "shape" in x, "symmetric_difference() operator can only process atomic shapes."

            if x["shape"] is None:
                if i == 0:
                    tmp_shape = self.shape_names
                else:
                    tmp_shape = []
            elif x["shape"].startswith("not_"):
                tmp_shape = set(self.shape_names) - {x["shape"][4:]}
            else:
                tmp_shape = x["shape"].split("|")

            if x["color"] is None:
                if i == 0:
                    tmp_color = colors
                else:
                    tmp_color = []
            elif x["color"].startswith("not_"):
                tmp_color = set(colors) - {x["color"][4:]}
            else:
                tmp_color = x["color"].split("|")

            if x["size"] is None:
                if i == 0:
                    tmp_size = sizes
                else:
                    tmp_size = []
            elif x["size"].startswith("not_"):
                tmp_size = set(sizes) - {x["size"][4:]}
            else:
                tmp_size = x["size"].split("|")

            if i == 0:
                out_shapes = set(tmp_shape)
                out_colors = set(tmp_color)
                out_sizes = set(tmp_size)
            else:
                out_shapes = out_shapes.symmetric_difference(set(tmp_shape))
                out_colors = out_colors.symmetric_difference(set(tmp_color))
                out_sizes = out_sizes.symmetric_difference(set(tmp_size))

        assert len(out_shapes) > 0, "symmetric_difference() produced no valid shape."
        assert len(out_colors) > 0, "symmetric_difference() produced no valid size."
        assert len(out_sizes) > 0, "symmetric_difference() produced no valid size."

        return {"shape": "|".join(out_shapes), "color": "|".join(out_colors), "size": "|".join(out_sizes)}

    # Prefetches symbols for the task. It performs rejection sampling to generate self.total_samples unique samples.
    # If it fails for self.patience successive trials, it gives up and the smaller set will be sampled with repetition.
    # The generated samples (whether they are enough or not) is finally randomly partitioned into train, validation and test splits.
    # Since injected noise is entirely perceptual, samples are considered unique based on their symbolic representation.
    def _prefetch_samples(self):
        self.dirty = True
        assert self.train_split > 0
        assert self.val_split > 0
        assert self.train_split + self.val_split < 1.0

        symbol_hash = set()

        i = 0
        # TODO: BUG!!! Rejection sampling done in this way follows the dataset balance because a 0.5 probability combined with the probability of having exhausted a sampling set skews the total probability towards the larger set.
        #       It is non-trivial to balance samples if one of the two sets is exhausted and the other is not, because we may loose diversity on the second one...
        while len(self.symbol_set) < self.total_samples and i < self.patience:
            label = self.rng.randint(0, 2)
            sample_set = "positive" if label == 1 else "negative"
            symbol = self._expand_symbol(self._sample_symbol(sample_set))

            if str(symbol) not in symbol_hash:  # Assuming no symbol is both positive and negative, otherwise the task is ill defined.
                if self.prolog_check:
                    query = "valid({})".format(self._symbol_to_prolog(symbol))

                    symbol_hash.add(str(symbol)) # We add the symbol to the hash set also in case it violates the rule, to avoid resampling it.

                    if self._check_rule(query) == bool(label):
                        i = 0

                        self.symbol_set.append((symbol, label))
                    else:
                        i += 1
                        self._log("{} sample {} {} the rule. Rejected.".format(("Positive" if label else "Negative"), str(symbol), ("violates" if label else "satisfies")), "warning")
                        self.rejected["positive" if label else "negative"]["rule"] += 1
                else:
                    symbol_hash.add(str(symbol))
                    self.symbol_set.append((symbol, label))
                    i = 0
            else:
                i += 1
                self.rejected["positive" if label else "negative"]["existing"] += 1

        if len(self.symbol_set) < self.total_samples:
            self._log(
                "Ran out of patience while generating samples (Requested: {}, generated: {}). Will sample with repetition.".format(
                    self.total_samples, len(self.symbol_set)), "warning")
            self.with_replacement = True
        else:
            self._log(
                "Samples generated successfully (Requested: {}, generated: {}). Will sample without repetition.".format(
                    self.total_samples, len(self.symbol_set)), "debug")
            self.with_replacement = False

        train_samples = int(np.ceil(self.train_split * len(self.symbol_set)))
        val_samples = int(np.ceil(self.val_split * len(self.symbol_set)))
        test_samples = len(self.symbol_set) - train_samples - val_samples

        train_req = int(np.ceil(self.train_split * self.total_samples))
        val_req = int(np.ceil(self.val_split * self.total_samples))
        test_req = self.total_samples - train_samples - val_samples

        assert train_samples > 0
        assert val_samples > 0
        assert test_samples > 0

        self.rng.shuffle(self.symbol_set)

        self.datasets["train"] = self.symbol_set[:train_samples]
        self.datasets["val"] = self.symbol_set[train_samples: train_samples + val_samples]
        self.datasets["test"] = self.symbol_set[train_samples + val_samples:]


        return train_samples, val_samples, test_samples, train_req, val_req, test_req

    # Outputs a supervised element (img, label, id, symbolic structure). The symbolic structure is a recursive combination of atomic objects, dictionaries and lists.
    def sample_supervised(self, split=None):
        self.dirty = True
        assert split is None or split in ["train", "val", "test"]

        if split is None:
            split = "train"

        if self.with_replacement:
            (symbol, label) = self.datasets[split][self.rng.choice(len(self.datasets[split]))]
        else:
            assert self.idx[split] < len(self.datasets[split]), "Index out of range for dataset {}.".format(split)

            (symbol, label) = self.datasets[split][self.idx[split]]
            self.idx[split] += 1

        sample_img = self._draw_sample(symbol)

        return sample_img, label, self.task_id, symbol


    # Produces a batch of supervised samples. If there are enough unique samples, the last batch may contain fewer elements (self.total_samples % batch_size).
    def get_batch(self, batch_size, split=None):
        assert split is None or split in ["train", "val", "test"]

        if split is None:
            split = "train"

        self.dirty = True

        if self.with_replacement: # If sampling is performed with replacement, it always outputs a full batch.
            out = [self.sample_supervised(split) for _ in range(batch_size)]
        else: # Otherwise, if the number of remaining samples is not enough to fill a batch, it outputs a smaller amount of them.
            out = []
            for i in range(min(batch_size, len(self.datasets[split]) - self.idx[split])):
                out.append(self.sample_supervised(split))

        return out

    # Teacher generator. It samples a decision and generates images accordingly.
    # The stream ends after self.total_samples * split_percentage samples, regardless of sampling mode (with or without replacement).
    # Supervisions are exponentially decayed with initial (i = 0) probability of self.gamma and final probability (i = self.total_samples * split_percentage) of self.beta.
    def get_stream(self, split=None):
        assert split is None or split in ["train", "val", "test"]
        if split is None:
            split = "train"

        self._log("BEGINNING TASK {} (Split: {}). Sampling is with{} replacement.".format(self.name, split, (
            "" if self.with_replacement else "out")))

        self.reset()
        self.dirty = True

        split_samples = self.requested_samples[split]

        self._log("Number of samples: {} (total), {} (current split), beta: {}, gamma: {}".format(self.total_samples,
                                                                                                  split_samples,
                                                                                                  self.beta, self.gamma))

        scale = (np.log(self.gamma / self.beta)) / (
                    self.gamma * split_samples)  # Guarantee a minimum probability of beta and a maximum of gamma. See appendix for an explanation.

        for i in range(split_samples):
            # Extract from an exponential distribution the decision of providing a supervised or unsupervised sample.
            # Each sample is guaranteed to be supervised with at least a probability of beta.
            t = i * scale

            supervised = int(self.rng.random() < self.gamma * np.exp(-self.gamma * t))
            sample_img, label, task_id, symbol = self.sample_supervised(split)
            self._log("{}SUPERVISED SAMPLE: {}".format(("" if supervised else "UN"), symbol))
            yield sample_img, label, supervised, task_id, symbol

        self._log("END OF TASK {}".format(self.name))

    # Resets the internal state of the task, including the random generator for reproducibility of results.
    def reset(self):
        if self.dirty:
            self.dirty = False
            self.rng.seed(self.seed)
            self.symbol_set = []
            self.datasets = {"train": [], "val": [], "test": []}
            self.with_replacement = True

            self.idx = {"train": 0, "val": 0, "test": 0}

            self.samples = {}
            self.requested_samples = {}
            self.rejected = {"positive": {"rule": 0, "existing": 0}, "negative": {"rule": 0, "existing": 0}}

            self.samples["train"], self.samples["val"], self.samples["test"], self.requested_samples["train"], self.requested_samples["val"], self.requested_samples["test"] = self._prefetch_samples()


# Curriculum generator class. It wraps multiple tasks into a single object.
class CurriculumGenerator:
    def __init__(self, config, curriculum, logger=None):
        self.tasks = []
        self.current_task = 0


        self.logger = logger
        self.config = {}

        # Hard-coded settings which would require rewriting functions.
        self.config["mandatory_keys"] = {"name": str, "beta": float, "gamma": float, "samples": int,
                                         "train_split": float, "val_split": float, "noisy_color": bool,
                                         "noisy_size": bool, "rot_noise": int, "positive_set": list, "negative_set": list}
        self.config["list_operators"] = {"sample": {"n": int}, "pick": {"n": int}, "first": {"n": int},
                                         "last": {"n": int},
                                         "permute": {}, "random_shift": {}, "shift": {"n": int}, "sort": {"order": str, "keys": list},
                                         "palindrome": {}, "mirror": {}, "repeat": {"n": int},
                                         "random_repeat": {"min": int, "max": int},
                                         "argsort": {"idx": list},
                                         "store": {"alias": str},
                                         "recall": {"alias": str}
                                         }  # key, params
        self.config["compositional_operators"] = ["in", "random", "stack", "side_by_side", "grid", "diag_ul_lr",
                                                  "diag_ll_ur", "stack_reduce_bb", "side_by_side_reduce_bb", "quadrant_ul", "quadrant_ur", "quadrant_ll", "quadrant_lr"]

        self.config["pregrounding_list_operators"] = {"sample_before": {"n": int}, "pick_before": {"n": int}, "first_before": {"n": int},
                                         "last_before": {"n": int},
                                         "permute_before": {}, "random_shift_before": {}, "shift_before": {"n": int},
                                         "palindrome_before": {}, "mirror_before": {}, "repeat_before": {"n": int},
                                         "random_repeat_before": {"min": int, "max": int},
                                         "union": {}, "intersection": {}, "difference": {}, "symmetric_difference": {},
                                                      "store_before": {"alias": str},
                                                      "any_composition": {}, "any_displacement": {}, "any_line": {},
                                                      "any_diag": {}, "any_non_diag": {}, "any_quadrant": {}, "quadrant_or_center": {}
                                         }  # key, params

        # Configurable settings.
        with open(config, "r") as file:
            try:
                self._parse_config(yaml.safe_load(file))
            except yaml.YAMLError as e:
                print(e)

        self.rng = np.random.RandomState(self.config["seed"])

        # Curriculum specifications.
        with open(curriculum, "r") as file:
            try:
                self._parse_curriculum(yaml.safe_load(file))
            except yaml.YAMLError as e:
                print(e)

    # Validates a configuration YAML file.
    # Colors and sizes are lists of single-key dictionaries, so that sort() can infer an order relation between entries.
    def validate_config(self, config):
        assert isinstance(config, dict), "The configuration must be a dictionary."

        assert "seed" in config, "Missing mandatory key seed."
        assert isinstance(config["seed"], int), "seed must be int, found {}.".format(type(config["seed"]))

        assert "canvas_size" in config, "Missing mandatory key canvas_size."
        assert isinstance(config["canvas_size"], list), "canvas_size must be a list, found {}.".format(
            type(config["canvas_size"]))
        assert len(config["canvas_size"]) == 2, "canvas_size must have exactly 2 elements, found {}.".format(
            len(config["canvas_size"]))
        assert isinstance(config["canvas_size"][0], int), "Canvas width (canvas_size[0]) must be int, found {}.".format(
            type(config["canvas_size"][0]))
        assert isinstance(config["canvas_size"][1],
                          int), "Canvas height (canvas_size[1]) must be int, found {}.".format(
            type(config["canvas_size"][1]))

        assert "padding" in config, "Missing mandatory key padding."
        assert isinstance(config["padding"], int), "padding must be int, found {}.".format(type(config["padding"]))
        assert config["padding"] >= 0, "padding must be non negative, found {}.".format(config["padding"])

        assert "size_noise" in config, "Missing mandatory key size_noise."
        assert isinstance(config["size_noise"], int), "size_noise must be int, found {}.".format(
            type(config["size_noise"]))
        assert config["size_noise"] >= 0, "size_noise must be non negative, found {}.".format(config["size_noise"])

        assert "h_sigma" in config, "Missing mandatory key h_sigma."
        assert isinstance(config["h_sigma"], float), "h_sigma must be float, found {}.".format(type(config["h_sigma"]))
        assert config["h_sigma"] >= 0, "h_sigma must be non negative, found {}.".format(config["h_sigma"])

        assert "s_sigma" in config, "Missing mandatory key s_sigma."
        assert isinstance(config["s_sigma"], float), "s_sigma must be float, found {}.".format(type(config["s_sigma"]))
        assert config["s_sigma"] >= 0, "s_sigma must be non negative, found {}.".format(config["s_sigma"])

        assert "v_sigma" in config, "Missing mandatory key v_sigma."
        assert isinstance(config["v_sigma"], float), "v_sigma must be float, found {}.".format(type(config["v_sigma"]))
        assert config["v_sigma"] >= 0, "v_sigma must be non negative, found {}.".format(config["v_sigma"])

        assert "bg_color" in config, "Missing mandatory key bg_color."
        assert isinstance(config["bg_color"], str), "s_sigma must be a string, found {}.".format(
            type(config["bg_color"]))
        assert re.match("^#[0-9a-f]{6}$", config["bg_color"],
                        flags=re.IGNORECASE) is not None, "bg_color must be an #rrggbb color, found {}.".format(
            config["bg_color"])

        assert "sizes" in config, "Missing mandatory key sizes."
        assert isinstance(config["sizes"], list), "sizes must be a list, found {}.".format(type(config["sizes"]))
        assert len(config["sizes"]) > 0, "sizes must have at least one element."
        for x in config["sizes"]:
            assert isinstance(x, dict), "Elements of sizes must be a dict, found {}.".format(type(x))
            assert len(x.keys()) == 1, "Elements of sizes must have a single key-value pair, found {}.".format(x)
            for k, v in x.items():
                assert isinstance(v, int), "Size {} must be int, found {}.".format(k, type(v))

        assert "colors" in config, "Missing mandatory key colors."
        assert isinstance(config["colors"], list), "colors must be a list, found {}.".format(type(config["colors"]))
        assert len(config["colors"]) > 0, "colors must have at least one element."
        for x in config["colors"]:
            assert isinstance(x, dict), "Elements of colors must be a dict, found {}.".format(type(x))
            assert len(x.keys()) == 1, "Elements of colors must have a single key-value pair, found {}.".format(x)
            for k, v in x.items():
                assert re.match("^#[0-9a-f]{6}$", v,
                                flags=re.IGNORECASE) is not None, "{} must be an #rrggbb color, found {}.".format(k, v)

        assert "shapes" in config, "Missing mandatory key shapes."
        assert isinstance(config["shapes"], list), "shapes must be a list, found {}.".format(type(config["shapes"]))
        assert len(config["shapes"]) > 0, "shapes must have at least one element."
        for x in config["shapes"]:
            assert isinstance(x, dict), "Elements of shapes must be a dict, found {}.".format(type(x))
            assert len(x.keys()) == 1, "Elements of shapes must have a single key-value pair, found {}.".format(x)
            for v in x.values():
                assert len(set(v.keys()).symmetric_difference(set(["type", "n", "rot"]))) == 0, "A shape must be characterized by: type, n, rot. Found {}.".format(x.values())
                assert v["type"] in ["polygon", "star", "ellipse"], "Allowed types for shapes are [polygon, star, ellipse], found {}.".format(v["type"])
                assert int(v["n"]) > 0, "n must be greater than 0. n = number of sides for polygons, n = number of points for stars, n = (integer) ratio between major and minor axis for ellipses."
                assert float(v["rot"]) >= 0 or float(v["rot"]) < 360, "rot must be between 0 (included) and 360 (excluded), found {}.".format(v["rot"])

        if "background_knowledge" in config:
            assert isinstance(config["background_knowledge"], str), "Optional parameter background_knowledge should be a string, {} found.".format(type(config["background_knowledge"]))
            assert os.path.isfile(config["background_knowledge"]), "File {} does not exist.".format(config["background_knowledge"])

    # Wrapper for recursive validation of a curriculum YAML file.
    def validate_curriculum(self, curriculum):
        assert isinstance(curriculum, list), "The curriculum must be a list of tasks."
        for i, t in enumerate(curriculum):
            assert isinstance(t, dict), "Task {} is not a dictionary.".format(i)

            for k, v in self.config["mandatory_keys"].items():
                assert k in t, "Missing mandatory key {}".format(k)
                assert isinstance(t[k], v), "Type mismatch: {} should be of type {}, found {}.".format(k, v, type(t[k]))

            assert len(t["positive_set"]) > 0, "The positive set list cannot be empty."
            assert len(t["negative_set"]) > 0, "The negative set list cannot be empty."

            self._recursive_validate(t["positive_set"])
            self._recursive_validate(t["negative_set"])

    # Recursively validates a sample set in the curriculum specification.
    def _recursive_validate(self, sample_set):
        if isinstance(sample_set, dict):  # Base case: atomic object.

            assert "shape" in sample_set, "Atomic object must have a shape, found {}.".format(sample_set)
            assert "color" in sample_set, "Atomic object must have a color, found {}.".format(sample_set)
            assert "size" in sample_set, "Atomic object must have a size, found {}.".format(sample_set)

            if sample_set["shape"] is None:  # Any value.
                ok = True
            elif sample_set["shape"].startswith("not_"):  # Negation.
                ok = sample_set["shape"][4:] in self.config["shape_names"]
            else:  # Disjunction or single value.
                tmp = sample_set["shape"].split("|")
                ok = True
                for i in tmp:
                    ok = ok and i in self.config["shape_names"]

            assert ok, "Invalid shape. Found {}.".format(sample_set["shape"])

            if sample_set["color"] is None:
                ok = True
            elif sample_set["color"].startswith("not_"):
                ok = sample_set["color"][4:] in [list(x.keys())[0] for x in self.config["colors"]]
            else:
                tmp = sample_set["color"].split("|")
                ok = True
                for i in tmp:
                    ok = ok and i in [list(x.keys())[0] for x in self.config["colors"]]

            assert ok, "Invalid color. Found {}.".format(sample_set["color"])

            if sample_set["size"] is None:
                ok = True
            elif sample_set["size"].startswith("not_"):
                ok = sample_set["size"][4:] in [list(x.keys())[0] for x in self.config["sizes"]]
            else:
                tmp = sample_set["size"].split("|")
                ok = True
                for i in tmp:
                    ok = ok and i in [list(x.keys())[0] for x in self.config["sizes"]]

            assert ok, "Invalid size. Found {}.".format(sample_set["size"])

        else:  # Recursive case: list.
            assert sample_set is not None, "Expected a list or a dict. Check your indentation."
            assert isinstance(sample_set, list), "Expected a list, found {}.".format(type(sample_set))
            for s in sample_set:
                assert isinstance(s, dict), "Expected a dict, found {}.".format(type(sample_set))

                comp_ops = list(set(s.keys()).intersection(self.config["compositional_operators"]))
                list_ops = list(set(s.keys()).intersection(self.config["list_operators"].keys()))
                prelist_ops = list(set(s.keys()).intersection(self.config["pregrounding_list_operators"].keys()))
                assert len(comp_ops) + len(
                    list_ops) <= 1, "There can be at most one operator at this level. Found {}".format(
                    comp_ops.union(list_ops))

                if len(comp_ops) + len(list_ops) + len(prelist_ops) == 0:
                    self._recursive_validate(s)
                elif len(comp_ops) == 1:
                    self._recursive_validate(s[comp_ops[0]])
                elif len(list_ops) == 1:
                    op = list_ops[0]
                    if len(self.config["list_operators"][op]) == 0:
                        self._recursive_validate(s[op])
                    else:
                        for k, v in self.config["list_operators"][op].items():
                            assert k in s[op], "Missing mandatory parameter {}.".format(k)
                            assert isinstance(s[op][k], v), "Expected parameter {} of type {}. Found {}.".format(k, v, type(s[op][k]))

                            if op != "recall":
                                assert "list" in s[op], "Missing mandatory parameter 'list'."
                                assert isinstance(s[op]["list"],
                                                  list), "Mandatory parameter 'list' must be a list. Found {}.".format(
                                    type(s[op]["list"]))
                                self._recursive_validate(s[op]["list"])
                elif len(prelist_ops) == 1:
                    op = prelist_ops[0]
                    if len(self.config["pregrounding_list_operators"][op]) == 0:
                        self._recursive_validate(s[op])
                    else:
                        for k, v in self.config["pregrounding_list_operators"][op].items():
                            assert s[op] is not None, "Expected a list or a dict. Check your indentation."
                            assert k in s[op], "Missing mandatory parameter {}.".format(k)
                            assert isinstance(s[op][k], v),\
                                "Expected parameter {} of type {}. Found {}.".format(k, v, type(s[op][k]))

                            assert "list" in s[op], "Missing mandatory parameter 'list'."
                            assert isinstance(s[op]["list"], list),\
                                "Mandatory parameter 'list' must be a list. Found {}.".format(type(s[op]["list"]))
                            self._recursive_validate(s[op]["list"])

    # Parse the global configuration YAML.
    def _parse_config(self, config):
        self.validate_config(config)

        self.config["seed"] = int(config["seed"])
        self.config["canvas_size"] = (max(32, int(config["canvas_size"][0])), max(32, int(config["canvas_size"][1])))
        self.config["padding"] = max(0, int(config["padding"]))
        self.config["bg_color"] = config["bg_color"]
        self.config["colors"] = config["colors"]
        self.config["sizes"] = config["sizes"]
        self.config["shapes"] = {list(x.keys())[0]: list(x.values())[0] for x in config["shapes"]} # For rendering.
        self.config["shape_names"] = [list(x.keys())[0] for x in config["shapes"]] # For validation and sorting.
        self.config["size_noise"] = int(config["size_noise"])
        self.config["h_sigma"] = max(0.0, float(config["h_sigma"]))
        self.config["s_sigma"] = max(0.0, float(config["s_sigma"]))
        self.config["v_sigma"] = max(0.0, float(config["v_sigma"]))
        if "background_knowledge" in config:
            self.config["background_knowledge"] = config["background_knowledge"]
            self.config["interpreter"] = pyswip.Prolog()
            self.config["interpreter"].consult(self.config["background_knowledge"])
        else:
            self.config["background_knowledge"] = None
            self.config["interpreter"] = None

    # Parse the curriculum YAML.
    def _parse_curriculum(self, curriculum):
        self.validate_curriculum(curriculum)

        for i, c in enumerate(curriculum):
            self.tasks.append(Task(i, self.config, c, logger=self.logger))

    # Reset the teacher and every task in the curriculum.
    def reset(self):
        self.current_task = 0
        self.rng.seed(self.config["seed"])

        for t in self.tasks:
            t.reset()


    # Generator for the entire curriculum. It visits each task in order and returns a batch. Optionally corrupts the task id.
    def generate_curriculum(self, split=None, task_id_noise=0.0, batch_size=1):

        assert split in ["train", "val", "test"] or split is None
        assert 0.0 <= task_id_noise and task_id_noise <= 1.0
        assert batch_size > 0

        if split is None:
            split = "train"

        self.reset()
        i = 0
        tid_np = np.zeros(batch_size, dtype=np.uint16)
        img_np = np.zeros((batch_size, self.config["canvas_size"][0], self.config["canvas_size"][1], 3), dtype=np.uint8)
        label_np = np.zeros(batch_size, dtype=bool)
        supervised_np = np.zeros(batch_size, dtype=bool)
        symbols_list = []

        while self.current_task < len(self.tasks):
            for sample in self.tasks[self.current_task].get_stream(split):

                sample_img, label, supervised, tid, symbol = sample

                tid_np[i] = tid
                img_np[i, :, :, :] = sample_img
                label_np[i] = label
                supervised_np[i] = supervised
                symbols_list.append(symbol)

                i += 1

                if i >= batch_size:
                    tid_np = np.where(self.rng.random(size=batch_size) < task_id_noise,
                                      self.rng.randint(len(self.tasks), size=batch_size), tid_np)

                    yield img_np, label_np, supervised_np, tid_np, symbols_list
                    i = 0
                    tid_np = np.zeros(batch_size, dtype=np.uint16)
                    img_np = np.zeros((batch_size, self.config["canvas_size"][0], self.config["canvas_size"][1], 3),
                                      dtype=np.uint8)
                    label_np = np.zeros(batch_size, dtype=bool)
                    supervised_np = np.zeros(batch_size, dtype=bool)
                    symbols_list = []

            self.current_task += 1

            if i > 0:
                tid_np[:i] = np.where(self.rng.random(size=i) < task_id_noise,
                                      self.rng.randint(len(self.tasks), size=i), tid_np[:i])
                yield img_np[:i, :, :, :], label_np[:i], supervised_np[:i], tid_np[:i], symbols_list

    # Generator for the entire curriculum. Works like generate_curriculum, but the task from which a sample is extracted is selected randomly.
    # Since each task will be exhausted at some point, sampling is not completely i.i.d., with batches later in the curriculum being sampled from fewer and fewer tasks.
    def generate_shuffled_curriculum(self, split=None, task_id_noise=0.0, batch_size=1):
        assert split in ["train", "val", "test"] or split is None
        assert 0.0 <= task_id_noise and task_id_noise <= 1.0
        assert batch_size > 0

        if split is None:
            split = "train"

        self.reset()
        i = 0
        task_iterators = [t.get_stream(split) for t in self.tasks]

        tid_np = np.zeros(batch_size, dtype=np.uint16)
        img_np = np.zeros((batch_size, self.config["canvas_size"][0], self.config["canvas_size"][1], 3), dtype=np.uint8)
        label_np = np.zeros(batch_size, dtype=bool)
        supervised_np = np.zeros(batch_size, dtype=bool)
        symbols_list = []

        while len(task_iterators) > 0:
            task = self.rng.choice(task_iterators)

            try:
                sample_img, label, supervised, tid, symbol = next(task)

                tid_np[i] = tid
                img_np[i, :, :, :] = sample_img
                label_np[i] = label
                supervised_np[i] = supervised
                symbols_list.append(symbol)

                i += 1

                if i >= batch_size:
                    tid_np = np.where(self.rng.random(size=batch_size) < task_id_noise,
                                      self.rng.randint(len(self.tasks), size=batch_size), tid_np)

                    yield img_np, label_np, supervised_np, tid_np, symbols_list
                    i = 0
                    tid_np = np.zeros(batch_size, dtype=np.uint16)
                    img_np = np.zeros((batch_size, self.config["canvas_size"][0], self.config["canvas_size"][1], 3),
                                      dtype=np.uint8)
                    label_np = np.zeros(batch_size, dtype=bool)
                    supervised_np = np.zeros(batch_size, dtype=bool)
                    symbols_list = []
            except StopIteration:
                task_iterators.remove(task)

        if i > 0:
            tid_np[:i] = np.where(self.rng.random(size=i) < task_id_noise,
                                  self.rng.randint(len(self.tasks), size=batch_size), tid_np)
            yield img_np[:i, :, :, :], label_np[:i], supervised_np[:i], tid_np[:i], symbols_list

    # Returns a batch for task i.
    def get_batch(self, i, batch_size, split=None):
        if i < len(self.tasks):
            return self.tasks[i].get_batch(batch_size, split)
        else:
            raise IndexError()

    # Generator for a specific task in the curriculum.
    def get_stream(self, i, split=None):
        if i < len(self.tasks):
            return self.tasks[i].get_stream(split)
        else:
            raise IndexError()

    # Returns a batch for the current task.
    def get_current_batch(self, batch_size, split=None):
        return self.get_batch(self.current_task, batch_size, split)


"""
Computing the scale value for the exponential distribution:

We cannot use directly a geometric distribution in virtue of the fact that the number of samples is large and probabilities quickly drop to zero.
We use an exponential distribution opportunely limited between 0 and a value such that the minimum probability is beta.

We obtain a continuous variable by normalizing task progression in [0,1]: t = i / self.total, then we rescale this value by:
scale = log(gamma / beta) / gamma.

This scale factor comes from solving the equation:
beta = gamma * e **(-gamma * scale)

This can be easily rewritten as:
log(beta) = log(gamma) - gamma * scale * log(e)
log(beta) = log(gamma) - gamma * scale
log(beta) - log(gamma) = -gamma * scale
(log(gamma) - log(beta)) / gamma = scale
log(gamma / beta) / gamma = scale

gamma * e ** (-gamma * scale * i / total) is guaranteed to have a maximum gamma in i=0 and a minimum beta in i=total.
"""