from manimlib.imports import *

from sys import maxsize

class Nixotica(Scene):
    def construct(self):
        name = TextMobject("Nixotica")
        name.scale(3)
        self.play(Write(name))
        self.wait()
        left = TextMobject("\guillemotleft")
        right = TextMobject("\guillemotright")
        left.scale(3)
        right.scale(3)
        left.next_to(name, direction = LEFT)
        right.next_to(name, direction = RIGHT)
        self.play(Write(left), Write(right))
        self.wait(3)
        self.play(FadeOut(name), FadeOut(left), FadeOut(right))

class ProposeTheorem(Scene):
    def construct(self):
        top = TextMobject("Does there exist an initial pivot point $P$")
        mid = TextMobject("in $S$ such that the union of all possible cycles")
        bot = TextMobject("starting from $P$ result in a completed digraph?")

        top.shift(UP)
        mid.next_to(top, direction=DOWN)
        bot.next_to(mid, direction=DOWN)

        self.play(Write(top))
        self.play(Write(mid))
        self.play(Write(bot))

class S3Complete(GraphScene):
    def construct(self):
        dots = []
        dot1 = Dot(point = ORIGIN + 3*UP)
        dot2 = Dot(point = ORIGIN + 3*LEFT + 3*DOWN)
        dot3 = Dot(point = ORIGIN + 3*RIGHT + 3*DOWN)
        dots.append(dot1)
        dots.append(dot2)
        dots.append(dot3)
        for i in range(len(dots)):
            self.play(ShowCreation(dots[i]))
        for i in range(len(dots)):
            for j in range(i + 1, len(dots)):
                arrow = DoubleArrow(dots[i].get_center(), dots[j].get_center())
                self.play(GrowArrow(arrow))

class S5Complete(GraphScene):
    def construct(self):
        dots = []
        dot1 = Dot(point = ORIGIN + 3*UP)
        dot2 = Dot(point = ORIGIN + 0.5*UP + 3*LEFT)
        dot3 = Dot(point = ORIGIN + 3*DOWN + 2*LEFT)
        dot4 = Dot(point = ORIGIN + 3*DOWN + 2*RIGHT)
        dot5 = Dot(point = ORIGIN + 0.5*UP + 3*RIGHT)
        dots.append(dot1)
        dots.append(dot2)
        dots.append(dot3)
        dots.append(dot4)
        dots.append(dot5)
        for i in range(len(dots)):
            self.play(ShowCreation(dots[i]))
        for i in range(len(dots)):
            for j in range(i + 1, len(dots)):
                arrow = DoubleArrow(dots[i].get_center(), dots[j].get_center(), stroke_width = 3, preserve_tip_size_when_scaling = False)
                self.play(GrowArrow(arrow))

class SnComplete(GraphScene):
    def construct(self):
        dots = []
        for i in range(20):
            x = random.uniform(-4, 4)
            y = random.uniform(-3, 3)
            dot = Dot(point = ORIGIN + x*RIGHT + y*UP)
            dots.append(dot)
            self.play(ShowCreation(dots[i]))
        
        for i in range(len(dots)):
            for j in range(i + 1, len(dots)):
                arrow = DoubleArrow(dots[i].get_center(), dots[j].get_center(), stroke_width = 3, preserve_tip_size_when_scaling = False)
                self.play(GrowArrow(arrow))

class WindmillScene(Scene):
    CONFIG = {
        "dot_config": {
            "fill_color": LIGHT_GREY,
            "radius": 0.05,
            "background_stroke_width": 2,
            "background_stroke_color": BLACK,
        },
        "windmill_style": {
            "stroke_color": RED,
            "stroke_width": 2,
            "background_stroke_width": 3,
            "background_stroke_color": BLACK,
        },
        "windmill_length": 2 * FRAME_WIDTH,
        "windmill_rotation_speed": 0.25,
        "hit_sound": "pen_click.wav",
        "leave_shadows": False,
        "draw_arrows": False
    }

    def get_random_point_set(self, n_points=11, width=6, height=6):
        return np.array([
            [
                -width / 2 + np.random.random() * width,
                -height / 2 + np.random.random() * height,
                0
            ]
            for n in range(n_points)
        ])

    def get_dots(self, points):
        return VGroup(*[
            Dot(point, **self.dot_config)
            for point in points
        ])

    def get_windmill(self, points, pivot=None, angle=TAU / 4):
        line = Line(LEFT, RIGHT)
        line.set_length(self.windmill_length)
        line.set_angle(angle)
        line.set_style(**self.windmill_style)

        line.point_set = points

        if pivot is not None:
            line.pivot = pivot
        else:
            line.pivot = points[0]

        line.rot_speed = self.windmill_rotation_speed

        line.add_updater(lambda l: l.move_to(l.pivot))
        return line

    def get_pivot_dot(self, windmill, color=YELLOW):
        pivot_dot = Dot(color=YELLOW)
        pivot_dot.add_updater(lambda d: d.move_to(windmill.pivot))
        return pivot_dot

    def start_leaving_shadows(self):
        self.leave_shadows = True
        self.add(self.get_windmill_shadows())

    def get_windmill_shadows(self):
        if not hasattr(self, "windmill_shadows"):
            self.windmill_shadows = VGroup()
        return self.windmill_shadows

    def next_pivot_and_angle(self, windmill):
        curr_angle = windmill.get_angle()
        pivot = windmill.pivot
        non_pivots = list(filter(
            lambda p: not np.all(p == pivot),
            windmill.point_set
        ))

        angles = np.array([
            -(angle_of_vector(point - pivot) - curr_angle) % PI
            for point in non_pivots
        ])

        # Edge case for 2 points
        tiny_indices = angles < 1e-6
        if np.all(tiny_indices):
            return non_pivots[0], PI

        angles[tiny_indices] = np.inf
        index = np.argmin(angles)
        return non_pivots[index], angles[index]

    def rotate_to_next_pivot(self, windmill, max_time=None, added_anims=None):
        """
        Returns animations to play following the contact, and total run time
        """
        new_pivot, angle = self.next_pivot_and_angle(windmill)
        change_pivot_at_end = True

        if added_anims is None:
            added_anims = []

        run_time = angle / windmill.rot_speed
        if max_time is not None and run_time > max_time:
            ratio = max_time / run_time
            rate_func = (lambda t: ratio * t)
            run_time = max_time
            change_pivot_at_end = False
        else:
            rate_func = linear

        for anim in added_anims:
            if anim.run_time > run_time:
                anim.run_time = run_time

        self.play(
            Rotate(
                windmill,
                -angle,
                rate_func=rate_func,
                run_time=run_time,
            ),
            *added_anims,
        )

        if change_pivot_at_end:
            self.handle_pivot_change(windmill, new_pivot)

        # Return animations to play
        return [self.get_hit_flash(new_pivot)], run_time

    def handle_pivot_change(self, windmill, new_pivot):
        if self.draw_arrows:
            arrow = Arrow(windmill.pivot, new_pivot)
            self.play(GrowArrow(arrow))

        windmill.pivot = new_pivot
        if self.leave_shadows:
            new_shadow = windmill.copy()
            new_shadow.fade(0.5)
            new_shadow.set_stroke(width=1)
            new_shadow.clear_updaters()
            shadows = self.get_windmill_shadows()
            shadows.add(new_shadow)

    def let_windmill_run(self, windmill, time):
        # start_time = self.get_time()
        # end_time = start_time + time
        # curr_time = start_time
        anims_from_last_hit = []
        while time > 0:
            anims_from_last_hit, last_run_time = self.rotate_to_next_pivot(
                windmill,
                max_time=time,
                added_anims=anims_from_last_hit,
            )
            time -= last_run_time
            # curr_time = self.get_time()

    def let_windmill_run_angle(self, windmill, angle=2*PI):
        anims_from_last_hit = []
        tot_angle = 0
        while tot_angle < angle:
            first_angle = windmill.get_angle()
            anims_from_last_hit, last_run_time = self.rotate_to_next_pivot(
                windmill,
                max_time=maxsize,
                added_anims=anims_from_last_hit,
            )
            second_angle = windmill.get_angle()
            tot_angle += abs(second_angle - first_angle)

    def let_windmills_run(self, windmills, time):
        anims_from_last_hit = []
        while time > 0:
            for windmill in windmills:
                anims_from_last_hit, last_run_time = self.rotate_to_next_pivot(
                    windmill,
                    max_time=time,
                    added_anims=anims_from_last_hit,
            )
            time -= last_run_time
    
    def add_dot_color_updater(self, dots, windmill, **kwargs):
        for dot in dots:
            dot.add_updater(lambda d: self.update_dot_color(
                d, windmill, **kwargs
            ))

    def update_dot_color(self, dot, windmill, color1=BLUE, color2=GREY_BROWN):
        perp = rotate_vector(windmill.get_vector(), TAU / 4)
        dot_product = np.dot(perp, dot.get_center() - windmill.pivot)
        if dot_product > 0:
            dot.set_color(color1)
        # elif dot_product < 0:
        else:
            dot.set_color(color2)
        # else:
        #     dot.set_color(WHITE)

        dot.set_stroke(
            # interpolate_color(dot.get_fill_color(), WHITE, 0.5),
            WHITE,
            width=2,
            background=True
        )

    def get_hit_flash(self, point):
        flash = Flash(
            point,
            line_length=0.1,
            flash_radius=0.2,
            run_time=0.5,
            remover=True,
        )
        flash_mob = flash.mobject
        for submob in flash_mob:
            submob.reverse_points()
        return Uncreate(
            flash.mobject,
            run_time=0.25,
            lag_ratio=0,
        )

    def get_pivot_counters(self, windmill, counter_height=0.25, buff=0.2, color=WHITE):
        points = windmill.point_set
        counters = VGroup()
        for point in points:
            counter = Integer(0)
            counter.set_color(color)
            counter.set_height(counter_height)
            counter.next_to(point, UP, buff=buff)
            counter.point = point
            counter.windmill = windmill
            counter.is_pivot = False
            counter.add_updater(self.update_counter)
            counters.add(counter)
        return counters

    def update_counter(self, counter):
        dist = get_norm(counter.point - counter.windmill.pivot)
        counter.will_be_pivot = (dist < 1e-6)
        if (not counter.is_pivot) and counter.will_be_pivot:
            counter.increment_value()
        counter.is_pivot = counter.will_be_pivot

    def get_orientation_arrows(self, windmill, n_tips=20):
        tips = VGroup(*[
            ArrowTip(start_angle=0)
            for x in range(n_tips)
        ])
        tips.stretch(0.75, 1)
        tips.scale(0.5)

        tips.rotate(windmill.get_angle())
        tips.match_color(windmill)
        tips.set_stroke(BLACK, 1, background=True)
        for tip, a in zip(tips, np.linspace(0, 1, n_tips)):
            tip.shift(
                windmill.point_from_proportion(a) - tip.points[0]
            )
        return tips

    def get_left_right_colorings(self, windmill, opacity=0.3):
        rects = VGroup(VMobject(), VMobject())
        rects.const_opacity = opacity

        def update_regions(rects):
            p0, p1 = windmill.get_start_and_end()
            v = p1 - p0
            vl = rotate_vector(v, 90 * DEGREES)
            vr = rotate_vector(v, -90 * DEGREES)
            p2 = p1 + vl
            p3 = p0 + vl
            p4 = p1 + vr
            p5 = p0 + vr
            rects[0].set_points_as_corners([p0, p1, p2, p3])
            rects[1].set_points_as_corners([p0, p1, p4, p5])
            rects.set_stroke(width=0)
            rects[0].set_fill(BLUE, rects.const_opacity)
            rects[1].set_fill(GREY_BROWN, rects.const_opacity)
            return rects

        rects.add_updater(update_regions)
        return rects

class S3CompleteWindmill(WindmillScene):
    CONFIG = {
        "n_points": 3,
        "run_time": 30,
        "random_seed": 0,
        "draw_arrows": True
    }

    def construct(self):
        points = np.array([[0, 3, 0], [-3, -3, 0], [3, -3, 0]])
        sorted_points = sorted(list(points), key=lambda p: p[1])

        dots = self.get_dots(points)
        windmill = self.get_windmill(points, sorted_points[0], angle=PI / 4)
        windmill2 = self.get_windmill(points, sorted_points[0], angle=PI / 2)
        pivot_dot = self.get_pivot_dot(windmill)
        # self.add_dot_color_updater(dots, windmill)

        self.add(windmill)
        self.add(windmill2)
        self.add(dots)
        self.add(pivot_dot)

        windmills = [windmill, windmill2]

        self.let_windmills_run(windmills, self.run_time)

class S3HullWindmill(WindmillScene):
    CONFIG = {
        "n_points": 3,
        "run_time": 30,
        "random_seed": 0,
        "draw_arrows": True
    }

    def construct(self):
        points = np.array([[0, 3, 0], [-3, -3, 0], [3, -3, 0]])
        sorted_points = sorted(list(points), key=lambda p: p[1])

        dots = self.get_dots(points)
        windmill = self.get_windmill(points, sorted_points[0], angle=PI / 2)
        pivot_dot = self.get_pivot_dot(windmill)

        self.add(windmill)
        self.add(dots)
        self.add(pivot_dot)

        self.let_windmill_run(windmill, self.run_time)

class S3InnerWindmill(WindmillScene):
    CONFIG = {
        "n_points": 3,
        "run_time": 30,
        "random_seed": 0,
        "draw_arrows": True
    }

    def construct(self):
        points = np.array([[0, 3, 0], [-3, -3, 0], [3, -3, 0]])
        sorted_points = sorted(list(points), key=lambda p: p[1])

        dots = self.get_dots(points)
        windmill = self.get_windmill(points, sorted_points[0], angle=PI / 4)
        pivot_dot = self.get_pivot_dot(windmill)

        self.add(windmill)
        self.add(dots)
        self.add(pivot_dot)

        self.let_windmill_run(windmill, self.run_time)

class WindmillExample(WindmillScene):
    CONFIG = {
        "n_points": 10,
        "random_seed": 0,
        "run_time": 30
    }

    def construct(self):
        points = self.get_random_point_set(self.n_points)
        points[:, 0] *= 1.5
        sorted_points = sorted(list(points), key=lambda p: p[1])
        sorted_points[4] += RIGHT

        dots = self.get_dots(points)
        windmill = self.get_windmill(points, sorted_points[5], angle=PI / 4)
        windmill_label = TextMobject("$\ell$")
        windmill_label.shift(LEFT+DOWN*0.7)

        pivot_dot = self.get_pivot_dot(windmill)
        pivot_label = TextMobject("$P$")
        pivot_label.next_to(pivot_dot)

        self.add(windmill)
        self.add(dots)
        self.add(pivot_dot)

        S_label = TextMobject("$S$")
        S_label.scale(2)
        S_label.to_corner(corner = UP + RIGHT)
        S_label.shift(LEFT*0.5)

        border = [(-5, -3.5, 0),
                  (-5, 3.5, 0),
                  (5, 3.5, 0),
                  (5, -3.5, 0)]

        S_box = Polygon(*border, color=GREEN)

        self.play(Write(windmill_label))
        self.play(Write(pivot_label))
        self.play(ShowCreation(S_box))
        self.play(Write(S_label))
        self.play(FadeOut(windmill_label))

        self.let_windmill_run(windmill, self.run_time)

class WindmillExample30Points(WindmillScene):
    CONFIG = {
        "n_points": 30,
        "random_seed": 0,
        "run_time": 30
    }

    def construct(self):
        points = self.get_random_point_set(self.n_points)
        points[:, 0] *= 1.5
        sorted_points = sorted(list(points), key=lambda p: p[1])
        sorted_points[4] += RIGHT

        dots = self.get_dots(points)
        windmill = self.get_windmill(points, sorted_points[5], angle=PI / 4)
        pivot_dot = self.get_pivot_dot(windmill)

        self.add(windmill)
        self.add(dots)
        self.add(pivot_dot)

        self.let_windmill_run(windmill, self.run_time)

class SetupS3(WindmillScene):
    CONFIG = {
        "n_points": 3,
        "windmill_rotation_speed": 0.5,
        "draw_arrows": True
    } 

    def construct(self):
        points = np.array([[0, 3, 0], [-3, -3, 0], [3, -3, 0]])
        sorted_points = sorted(list(points), key=lambda p: p[1])

        dots = self.get_dots(points)
        windmill = self.get_windmill(points, sorted_points[0], angle=PI/2)
        pivot_dot = self.get_pivot_dot(windmill)

        for dot in dots:
            self.play(ShowCreation(dot))

        pivot_label = TextMobject("$P$")
        pivot_label.next_to(pivot_dot)
        self.play(Write(pivot_label))

        self.play(ShowCreation(windmill))

        self.play(Rotate(windmill, angle=-1*PI/4))

        self.wait()
        
        self.play(Rotate(windmill, angle=-1*(PI/4+PI/32)))

        self.wait()

        self.play(Rotate(windmill, angle=-1*(PI-(PI/4+PI/4+PI/32))))

        self.wait()

        self.add(windmill)
        self.add(dots)
        self.add(pivot_dot)

        #self.let_windmill_run(windmill, 10)

class TwoConfigurations(Scene):
    def construct(self):
        threepoints = TextMobject("3 points")
        onepivot = TextMobject("1 pivot")
        line = Line(start=LEFT, end=RIGHT)
        twoconfig = TextMobject("2 configurations")
        
        onepivot.next_to(threepoints, direction=DOWN)
        line.next_to(onepivot, direction=DOWN)
        twoconfig.next_to(line, direction=DOWN)

        self.play(Write(threepoints))
        self.play(Write(onepivot))
        self.play(ShowCreation(line))
        self.play(Write(twoconfig))

class ArbitraryConvexHull(WindmillScene):
    CONFIG = {
        "n_points": 6,
        "random_seed": 2020,
        "run_time": 30,
        "windmill_rotation_speed": 0.5,
        "draw_arrows": True
    }

    def construct(self):
        points = np.array(
                 [[-4, 2, 0],
                  [0, 3, 0],
                  [2, 3, 0],
                  [4, 0, 0],
                  [3, -2, 0],
                  [-3, -3, 0]])
        sorted_points = sorted(list(points), key=lambda p: p[1])

        dots = self.get_dots(points)
        windmill = self.get_windmill(points, sorted_points[0], angle=3*PI/4)
        pivot_dot = self.get_pivot_dot(windmill)

        self.add(windmill)
        self.add(dots)
        self.add(pivot_dot)

        S_label = TextMobject("$S_n$")
        S_label.scale(2)
        S_label.to_corner(corner = UP + RIGHT)

        border = [(-5, -3.5, 0),
                  (-5, 3.5, 0),
                  (5, 3.5, 0),
                  (5, -3.5, 0)]

        S_box = Polygon(*border, color=GREEN)

        self.play(ShowCreation(S_box))
        self.play(ShowCreation(S_label))

        inner_points = points*0.75
        inner_hull = Polygon(*inner_points, color=WHITE, fill_color=WHITE, fill_opacity=0.1)
        self.play(ShowCreation(inner_hull))

        inner_label = TextMobject("$n-n_{hull}$ points")
        self.play(Write(inner_label))

        pivot_label = TextMobject("$P$")
        pivot_label.next_to(pivot_dot, direction=LEFT)
        self.play(Write(pivot_label))

        self.let_windmill_run(windmill, 13)

        self.wait(1)

        self.play(FadeOut(inner_hull), FadeOut(inner_label))

        self.wait(1)

        m = TextMobject("$m$")
        m.next_to(pivot_dot.get_center(), direction=UP)
        m.shift(LEFT + UP*math.sin(windmill.get_angle()) + RIGHT*math.cos(windmill.get_angle()))

        nminusm = TextMobject("$n-m$")
        nminusm.next_to(inner_label, direction=UP)

        self.play(Write(m))
        self.wait(1)
        self.play(Write(nminusm))

class LOnHull(WindmillScene):
    def construct(self):
        points = np.array([[-3, -3, 0]])
        windmill = self.get_windmill(points, points[0], angle=3*PI/4)
        pivot_dot = self.get_pivot_dot(windmill)

        S_label = TextMobject("$S_n$")
        S_label.scale(2)
        S_label.to_corner(corner = UP + RIGHT)

        border = [(-5, -3.5, 0),
                  (-5, 3.5, 0),
                  (5, 3.5, 0),
                  (5, -3.5, 0)]

        S_box = Polygon(*border, color=GREEN)

        self.play(ShowCreation(S_box))
        self.play(ShowCreation(S_label))

        pivot_label = TextMobject("$P$")
        pivot_label.next_to(pivot_dot, direction=LEFT)
        self.play(ShowCreation(pivot_dot))
        self.play(Write(pivot_label))

        self.play(ShowCreation(windmill))

        l_label = TextMobject("$\ell$")
        l_label.next_to(pivot_dot, direction=LEFT)
        l_label.shift(UP + LEFT)
        self.play(Write(l_label))

        inner_points = np.array([[-3, -3, 0],
                                 [-2, 3, 0],
                                 [1, 2, 0],
                                 [3, -2, 0],
                                 [0, -3, 0]])
        hull = Polygon(*inner_points, color=WHITE, fill_color=WHITE, fill_opacity=0.1)
        self.play(ShowCreation(hull))

        hull_label = TextMobject("Convex Hull")
        self.play(Write(hull_label))

        self.play(FadeOut(pivot_label), FadeOut(hull_label))

        arc = Arc(start_angle=windmill.get_angle(), angle=PI, radius=0.25, arc_center=pivot_dot.get_center())
        oneeighty = TextMobject("$180^{\circ}$")
        oneeighty.next_to(pivot_dot, direction=LEFT)
        self.play(Write(oneeighty), ShowCreation(arc))

        p1 = Point(location=RIGHT*-2+UP*3)
        p2 = Point(location=UP*-3)
        l1 = DashedLine(start=pivot_dot.get_center(), end=p1.get_center())
        l2 = DashedLine(start=pivot_dot.get_center(), end=p2.get_center())

        ang = l1.get_angle() - l2.get_angle()
        general_ang = TextMobject("$<180^{\circ}$")
        general_ang.next_to(pivot_dot, direction=UP*0.5+RIGHT)
        arc2 = Arc(start_angle=l2.get_angle(), angle=ang, radius=0.25, arc_center=pivot_dot.get_center())
        self.play(Write(general_ang), ShowCreation(arc2))

        self.wait(1)

        self.play(FadeOut(general_ang),
                  FadeOut(oneeighty),
                  FadeOut(arc),
                  FadeOut(arc2),
                  FadeOut(pivot_dot),
                  FadeOut(windmill),
                  FadeOut(l_label))
        
        inner_point = Dot(point=[0, 0, 0], color=YELLOW)
        inner_label = TextMobject("Inner Pivot").next_to(inner_point, direction=DOWN)
        self.play(ShowCreation(inner_point), Write(inner_label))

        self.wait(1)

        self.play(FadeOut(inner_label))

        p0 = Point(location=[-3, -3, 0])
        p1 = Point(location=[1, 2, 0])
        p2 = Point(location=[0, -3, 0])

        inner_border = np.array([p0.get_center(), p1.get_center(), p2.get_center()])
        inner_poly = Polygon(*inner_border, color=RED, fill_color=RED, fill_opacity=0.1)

        self.play(ShowCreation(inner_poly))

        windmill = self.get_windmill(points, inner_point, angle=3*PI/4)
        pivot_dot = self.get_pivot_dot(windmill)

        self.wait(1)
        self.play(ShowCreation(windmill))

        l_label.next_to(pivot_dot, direction=3*LEFT+UP)
        self.play(Write(l_label))
        self.wait(1)

        arrows = []
        for i in range(0, len(inner_points)):
            p0 = inner_points[i-1]*1.1
            p1 = inner_points[i]*1.1
            arrow = Arrow(p0, p1, color=RED)
            arrows.append(arrow)
            self.play(GrowArrow(arrows[i]))

        self.wait(1)
        self.play(FadeOut(l_label), FadeOut(windmill), FadeOut(inner_poly), 
                  FadeOut(arrows[0]), FadeOut(arrows[1]),
                  FadeOut(arrows[2]), FadeOut(arrows[3]),
                  FadeOut(arrows[4]), FadeOut(inner_point))
        self.wait(1)

        p1 = Dot(point=[-3, -3, 0])
        p1_label = TextMobject("$P_{pivot}$").next_to(p1, direction=LEFT)
        self.play(ShowCreation(p1), Write(p1_label))

        p2 = Dot(point=[-2, 3, 0])
        p2_label = TextMobject("$P_2$").next_to(p2, direction=LEFT)
        self.play(ShowCreation(p2), Write(p2_label))

        points = np.array([p1.get_center(), p2.get_center()])
        windmill = self.get_windmill(points, p1, angle=PI/4)

        self.play(ShowCreation(windmill))
        self.wait(1)

        m = TextMobject("$m$").next_to(p1.get_center(), direction=UP+RIGHT*1.5).set_color(RED)
        m.shift(0.75*LEFT + UP*math.sin(windmill.get_angle()) + RIGHT*math.cos(windmill.get_angle()))
        self.play(Write(m))
        self.wait(1)

        nminusm = TextMobject("$n-m$").next_to(p1.get_center(),direction=UP+RIGHT).set_color(RED)
        nminusm.shift(0.5*RIGHT + UP*math.sin(windmill.get_angle()) + RIGHT*math.cos(windmill.get_angle()))
        self.play(Write(nminusm))
        self.wait(1)

        windmill2 = self.get_windmill(points, p2, angle=3*PI/4).set_color(GREEN)
        self.play(ShowCreation(windmill2))
        self.wait(1)

        m2 = TextMobject("$m$").next_to(p2.get_center(), direction=DOWN+RIGHT).set_color(GREEN)
        m2.shift(UP + DOWN*math.sin(windmill.get_angle()) + RIGHT*math.cos(windmill.get_angle()))
        self.play(Write(m2))
        self.wait(1)

        nminusm2 = TextMobject("$n-m$").next_to(p2.get_center(),direction=DOWN).set_color(GREEN)
        nminusm2.shift(DOWN + DOWN*math.sin(windmill.get_angle()) + RIGHT*math.cos(windmill.get_angle()))
        self.play(Write(nminusm2))
        self.wait(1)

        self.play(FadeOut(m), FadeOut(nminusm), FadeOut(nminusm2), FadeOut(m2), FadeOut(windmill), FadeOut(windmill2))

        dots = []
        dot1 = Dot(point = ORIGIN + 0.5*RIGHT + DOWN)
        dot2 = Dot(point = ORIGIN + RIGHT + 0.5*UP)
        dot3 = Dot(point = ORIGIN + LEFT + 0.5*DOWN)
        dots.append(dot1)
        dots.append(dot2)
        dots.append(dot3)
        for i in range(len(dots)):
            self.play(ShowCreation(dots[i]))
        visarrows = []
        for i in range(len(dots)):
            arrow1 = Arrow(p1.get_center(), dots[i].get_center())
            arrow2 = Arrow(p2.get_center(), dots[i].get_center())
            self.play(GrowArrow(arrow1))            
            self.play(GrowArrow(arrow2))
            visarrows.append(arrow1)
            visarrows.append(arrow2)

        self.wait(1)
        self.play(FadeOut(visarrows[0]), FadeOut(visarrows[1]), FadeOut(visarrows[2]), FadeOut(visarrows[3]), FadeOut(visarrows[4]), FadeOut(visarrows[5]))
        for i in range(len(dots)):
            arrow1 = Arrow(dots[i].get_center(), p1.get_center(), color=RED, opacity=0.2)
            arrow2 = Arrow(dots[i].get_center(), p2.get_center(), color=RED, opacity=0.2)
            self.play(GrowArrow(arrow1))
            self.play(GrowArrow(arrow2))

        self.wait(1)

class HullToInsideComplete(GraphScene):
    def construct(self):
        inner_points = np.array([[1 , 2, 0],
                                 [-2, -2, 0],
                                 [5, -3, 0]])
        hull = Polygon(*inner_points, color=WHITE, fill_color=WHITE, fill_opacity=0.1)
        self.play(ShowCreation(hull))

        hulldots = []
        dot1 = Dot(point = ORIGIN + 2*UP + RIGHT)
        dot2 = Dot(point = ORIGIN + 2*LEFT + 2*DOWN)
        dot3 = Dot(point = ORIGIN + 5*RIGHT + 3*DOWN)
        hulldots.append(dot1)
        hulldots.append(dot2)
        hulldots.append(dot3)

        innerdots = []
        dot1 = Dot(point = ORIGIN + 1*RIGHT)
        dot2 = Dot(point = ORIGIN + 1.5*DOWN + 1.5*RIGHT)
        innerdots.append(dot1)
        innerdots.append(dot2)

        for i in range(len(hulldots)):
            self.play(ShowCreation(hulldots[i]))
        for i in range(len(innerdots)):
            self.play(ShowCreation(innerdots[i]))

        self.wait(1)

        accounted = TextMobject("Accounted for:")
        accounted.to_corner(corner = UP + LEFT)
        self.play(Write(accounted))
    
        perms = TextMobject("$$(P_{hull}, P_{inner})$$")
        perms.next_to(accounted, direction=DOWN)
        self.play(Write(perms))

        self.wait(1)

        arrows = []
        for i in range(len(hulldots)):
            for j in range(len(innerdots)):
                arrow = Arrow(hulldots[i].get_center(), innerdots[j].get_center())
                arrows.append(arrow)
                self.play(GrowArrow(arrow))

        self.wait(1)

        self.play(FadeOut(arrows[0]), FadeOut(arrows[1]), FadeOut(arrows[2]), FadeOut(arrows[3]), FadeOut(arrows[4]), FadeOut(arrows[5]))

        self.wait(1)

        notaccounted = TextMobject("Not accounted for:").set_color(RED)
        notaccounted.to_edge(edge=LEFT)
        self.play(Write(notaccounted))
    
        notperms = TextMobject("$$(P_{inner}, P_{hull})$$").set_color(RED)
        notperms.next_to(notaccounted, direction=DOWN)
        self.play(Write(notperms))

        self.wait(1)

        for i in range(len(innerdots)):
            for j in range(len(hulldots)):
                arrow = Arrow(innerdots[i].get_center(), hulldots[j].get_center()).set_color(RED)
                self.play(GrowArrow(arrow))

        self.wait(1)

class InnerHullPNotIntersect(WindmillScene):
    def construct(self):
        points = np.array([[0 , 3, 0],
                        [3, -3, 0],
                        [-3, -3, 0],
                        [0, 1.5, 0],
                        [1.5, -1.5, 0],
                        [-1.5, -1.5, 0],
                        [0, 0, 0]])

        dots = self.get_dots(points)

        for i in range(len(dots)):
            self.play(ShowCreation(dots[i]))

        self.wait(1)

        inner_point = [0, 0, 0]
        outer_point = [-1.5, -1.5, 0]
        P_label = TextMobject("$P$").next_to(inner_point, direction=DOWN)
        self.play(Write(P_label))

        self.wait(1)

        outer_windmill = self.get_windmill(points, outer_point, angle=3*PI/4)
        outer_pivot_dot = self.get_pivot_dot(outer_windmill)

        outer_label = TextMobject("$\ell$").next_to(outer_pivot_dot).shift(DOWN+RIGHT)
        self.play(ShowCreation(outer_windmill), Write(outer_label))

        inner_windmill = self.get_windmill(points, inner_point, angle=0)
        inner_pivot_dot = self.get_pivot_dot(inner_windmill)

        inner_label = TextMobject("$\ell'$").next_to(inner_pivot_dot, direction=UP).shift(RIGHT)
        self.play(ShowCreation(inner_windmill), Write(inner_label))

        self.wait(1)

        n1 = TextMobject("$n=1$").next_to(outer_pivot_dot, direction=LEFT)
        self.play(Write(n1))

        self.wait(1)
        
        n2 = TextMobject("$n=2$").next_to(inner_pivot_dot, direction=UP+LEFT)
        self.play(Write(n2))

        self.play(FadeOut(n1), FadeOut(n2), FadeOut(inner_label), FadeOut(outer_label))
        
        self.wait(1)

        self.add(outer_windmill)
        self.add(dots)

        self.let_windmill_run(outer_windmill, 9.45)

        self.wait(1)

        n1 = TextMobject("$n=1$").next_to([0, 1.5, 0], direction=UP).shift(RIGHT*2)
        n2.next_to(inner_pivot_dot, direction=UP).shift(RIGHT*2)

        self.play(Write(n1), Write(n2))

        self.wait(1)

        outer_hull = np.array([[-3, -3, 0],
                                 [0, 3, 0],
                                 [3, -3, 0]])
        outer_hull = Polygon(*outer_hull, color=WHITE, fill_color=WHITE, fill_opacity=0.1)
        self.play(ShowCreation(outer_hull))

        inner_hull = np.array([[-1.5, -1.5, 0],
                               [0, 1.5, 0],
                               [1.5, -1.5, 0]])
        inner_hull = Polygon(*inner_hull, color=GREEN, fill_color=GREEN, fill_opacity=0.1)
        self.play(ShowCreation(inner_hull))

        self.wait(1)

        self.wait(1)

        self.play(FadeOut(outer_hull), FadeOut(inner_hull), FadeOut(n1), FadeOut(n2), FadeOut(inner_windmill), FadeOut(outer_windmill), FadeOut(P_label))

class InnerAllProof(WindmillScene):
    def construct(self):
        points = np.array([[-1.5 , 2, 0],
                          [-3.5, -2, 0],
                          [4.5, -3, 0],
                          [-1, 0, 0],
                          [0.5, -1, 0]])
        
        dots = self.get_dots(points)

        for i in range(len(dots)):
            self.play(ShowCreation(dots[i]))

        early_windmill = self.get_windmill(points, points[1], angle=PI/6)
        early_pivot_dot = self.get_pivot_dot(early_windmill)

        late_windmill = self.get_windmill(points, points[0], angle=3*PI/5)
        late_pivot_dot = self.get_pivot_dot(late_windmill)

        P_inner = TextMobject("$P_{inner}$").next_to(points[4]).set_color(RED)
        self.play(Write(P_inner))

        self.wait(1)

        self.play(ShowCreation(early_windmill))

        early_label = TextMobject("$\ell_0$").next_to(early_pivot_dot, direction=LEFT).shift(LEFT).set_color(RED)
        self.play(Write(early_label))

        self.wait(1)

        n1 = TextMobject("$n=2$").next_to(early_pivot_dot, direction=UP).shift(UP).set_color(RED)
        m1 = TextMobject("$m=2$").next_to(early_pivot_dot, direction=DOWN).shift(DOWN).set_color(RED)
        self.play(Write(n1), Write(m1))

        self.wait(1)

        P2 = TextMobject("$P_2$").next_to(points[3], direction=LEFT).set_color(GREEN)
        self.play(Write(P2))

        self.play(ShowCreation(late_windmill))

        late_label = TextMobject("$\ell_1$").next_to(late_pivot_dot, direction=RIGHT).set_color(GREEN)
        self.play(Write(late_label))

        self.wait(1)

        n2 = TextMobject("$n=2$").next_to(late_pivot_dot, direction=RIGHT).shift(UP).set_color(GREEN)
        m2 = TextMobject("$m=2$").next_to(late_pivot_dot, direction=LEFT).shift(DOWN).set_color(GREEN)
        self.play(Write(n2), Write(m2))

        self.wait(1)

        self.play(FadeOut(n1), FadeOut(n2), FadeOut(m1), FadeOut(m2))

        implication = TextMobject("$180^{\circ}$ turn $\Rightarrow$").to_corner(corner = UP+ LEFT)
        implication2 = TextMobject("$\ell_0 = \ell_1$ at some point.").to_corner(corner = UP + LEFT).shift(DOWN*0.5)
        self.play(Write(implication))
        self.play(Write(implication2))

        self.wait(1)

        self.play(FadeOut(implication), FadeOut(implication2), FadeOut(early_label), FadeOut(late_label))

        self.let_windmill_run(early_windmill, 7.11)
        
        self.wait(1)

        self.play(FadeOut(early_windmill), FadeOut(late_windmill), FadeOut(P2))
        self.wait(1)

        P_Hull = TextMobject("$P_{hull}$").next_to(points[1], direction=RIGHT).set_color(BLUE)
        self.play(Write(P_Hull))

        early_windmill = self.get_windmill(points, points[1], angle=PI/3).set_color(BLUE)
        self.play(ShowCreation(early_windmill))

        n1 = TextMobject("$n=1$").next_to(early_pivot_dot, direction=UP).shift(UP).set_color(BLUE)
        m1 = TextMobject("$m=3$").next_to(early_pivot_dot, direction=DOWN).shift(DOWN+RIGHT).set_color(BLUE)

        self.play(Write(n1), Write(m1))

        self.wait(1)

        late_windmill = self.get_windmill(points, points[4], angle=PI/6)
        late_pivot_dot = self.get_pivot_dot(late_windmill)

        self.play(ShowCreation(late_windmill))
        self.wait(1)

        n2 = TextMobject("$n=1$").next_to(late_pivot_dot, direction=UP).shift(UP).set_color(RED)
        m2 = TextMobject("$m=3$").next_to(late_pivot_dot, direction=DOWN).shift(DOWN).set_color(RED)

        self.play(Write(n2), Write(m2))
        self.wait(1)

        self.play(FadeOut(n1), FadeOut(n2), FadeOut(m1), FadeOut(m2))

        self.let_windmill_run(early_windmill, 14.65)

        self.wait(1)

        self.play(FadeOut(P_Hull), FadeOut(P_inner), FadeOut(early_windmill), FadeOut(late_windmill))

        comb = DoubleArrow(early_pivot_dot.get_center(), late_pivot_dot.get_center()).set_color(YELLOW)
        self.play(ShowCreation(comb))

        exp = TextMobject("Both combinations accounted for.").next_to(late_pivot_dot, direction=DOWN).shift(DOWN).set_color(YELLOW)
        self.play(Write(exp))

        self.wait(1)

class Complete1(WindmillScene):
    CONFIG = {
        "draw_arrows": "True"
    }
    def construct(self):
        border = [(-5, -3.5, 0),
                    (-5, 3.5, 0),
                    (5, 3.5, 0),
                    (5, -3.5, 0)]

        S_box = Polygon(*border, color=GREEN)
        S_label = TextMobject("$S_1$").scale(2).to_corner(corner=UP+RIGHT).shift(LEFT*0.5)

        self.play(ShowCreation(S_box), Write(S_label))
        
        points = np.array([[0 , 3, 0],
                          [-3, -2, 0],
                          [0, 0, 0],
                          [1, 2, 0],
                          [-2, 2, 0],
                          [3, -1, 0]])
        
        dots = self.get_dots(points)

        for i in range(len(dots)):
            self.play(ShowCreation(dots[i]))

        windmill = self.get_windmill(points, points[1], angle=PI/2)
        pivot_dot = self.get_pivot_dot(windmill)

        P_init = TextMobject("$P_0$").next_to(pivot_dot, direction=LEFT)

        self.play(Write(P_init))

        self.wait(3)

        self.play(ShowCreation(windmill))

        self.let_windmill_run_angle(windmill, 2*PI)

class Complete2(WindmillScene):
    CONFIG = {
        "draw_arrows": "True"
    }
    def construct(self):
        border = [(-5, -3.5, 0),
                    (-5, 3.5, 0),
                    (5, 3.5, 0),
                    (5, -3.5, 0)]

        S_box = Polygon(*border, color=GREEN)
        S_label = TextMobject("$S_2$").scale(2).to_corner(corner=UP+RIGHT).shift(LEFT*0.5)

        self.play(ShowCreation(S_box), Write(S_label))
        
        points = np.array([[0 , 3, 0],
                          [-3, -2, 0],
                          [0, 0, 0],
                          [1, 2, 0],
                          [-2, 2, 0],
                          [3, -1, 0]])
        
        dots = self.get_dots(points)

        for i in range(len(dots)):
            self.play(ShowCreation(dots[i]))

        windmill = self.get_windmill(points, points[1], angle=PI/3)
        pivot_dot = self.get_pivot_dot(windmill)

        P_init = TextMobject("$P_0$").next_to(pivot_dot, direction=LEFT)

        self.play(Write(P_init))

        self.wait(3)

        self.play(ShowCreation(windmill))

        self.let_windmill_run_angle(windmill, 4*PI)

class Complete3(WindmillScene):
    CONFIG = {
        "draw_arrows": "True"
    }
    def construct(self):
        border = [(-5, -3.5, 0),
                    (-5, 3.5, 0),
                    (5, 3.5, 0),
                    (5, -3.5, 0)]

        S_box = Polygon(*border, color=GREEN)
        S_label = TextMobject("$S_3$").scale(2).to_corner(corner=UP+RIGHT).shift(LEFT*0.5)

        self.play(ShowCreation(S_box), Write(S_label))
        
        points = np.array([[0 , 3, 0],
                          [-3, -2, 0],
                          [0, 0, 0],
                          [1, 2, 0],
                          [-2, 2, 0],
                          [3, -1, 0]])
        
        dots = self.get_dots(points)

        for i in range(len(dots)):
            self.play(ShowCreation(dots[i]))

        windmill = self.get_windmill(points, points[1], angle=2*PI/7)
        pivot_dot = self.get_pivot_dot(windmill)

        P_init = TextMobject("$P_0$").next_to(pivot_dot, direction=LEFT)

        self.play(Write(P_init))

        self.wait(3)

        self.play(ShowCreation(windmill))

        self.let_windmill_run_angle(windmill, 4*PI)

class Complete4(WindmillScene):
    CONFIG = {
        "draw_arrows": "True"
    }
    def construct(self):
        border = [(-5, -3.5, 0),
                    (-5, 3.5, 0),
                    (5, 3.5, 0),
                    (5, -3.5, 0)]

        S_box = Polygon(*border, color=GREEN)
        S_label = TextMobject("$S_4$").scale(2).to_corner(corner=UP+RIGHT).shift(LEFT*0.5)

        self.play(ShowCreation(S_box), Write(S_label))
        
        points = np.array([[0 , 3, 0],
                          [-3, -2, 0],
                          [0, 0, 0],
                          [1, 2, 0],
                          [-2, 2, 0],
                          [3, -1, 0]])
        
        dots = self.get_dots(points)

        for i in range(len(dots)):
            self.play(ShowCreation(dots[i]))

        windmill = self.get_windmill(points, points[1], angle=PI/4-0.1)
        pivot_dot = self.get_pivot_dot(windmill)

        P_init = TextMobject("$P_0$").next_to(pivot_dot, direction=LEFT)

        self.play(Write(P_init))

        self.wait(3)

        self.play(ShowCreation(windmill))

        self.let_windmill_run_angle(windmill, 4*PI)

class Complete5(WindmillScene):
    CONFIG = {
        "draw_arrows": "True"
    }
    def construct(self):
        border = [(-5, -3.5, 0),
                    (-5, 3.5, 0),
                    (5, 3.5, 0),
                    (5, -3.5, 0)]

        S_box = Polygon(*border, color=GREEN)
        S_label = TextMobject("$S_5$").scale(2).to_corner(corner=UP+RIGHT).shift(LEFT*0.5)

        self.play(ShowCreation(S_box), Write(S_label))
        
        points = np.array([[0 , 3, 0],
                          [-3, -2, 0],
                          [0, 0, 0],
                          [1, 2, 0],
                          [-2, 2, 0],
                          [3, -1, 0]])
        
        dots = self.get_dots(points)

        for i in range(len(dots)):
            self.play(ShowCreation(dots[i]))

        windmill = self.get_windmill(points, points[1], angle=PI/6)
        pivot_dot = self.get_pivot_dot(windmill)

        P_init = TextMobject("$P_0$").next_to(pivot_dot, direction=LEFT)

        self.play(Write(P_init))

        self.wait(3)

        self.play(ShowCreation(windmill))

        self.let_windmill_run_angle(windmill, 4*PI)

class CompleteFinal(GraphScene):
    def construct(self):
        dots = []
        dot1 = Dot(point = ORIGIN + 3*UP)
        dot2 = Dot(point = ORIGIN + 3*LEFT + 2*DOWN)
        dot3 = Dot(point = ORIGIN)
        dot4 = Dot(point = ORIGIN + RIGHT + 2*UP)
        dot5 = Dot(point = ORIGIN + 2*LEFT + 2*UP)
        dot6 = Dot(point = ORIGIN + 3*RIGHT + DOWN)
        dots.append(dot1)
        dots.append(dot2)
        dots.append(dot3)
        dots.append(dot4)
        dots.append(dot5)
        dots.append(dot6)
        for i in range(len(dots)):
            self.play(ShowCreation(dots[i]))
        for i in range(len(dots)):
            for j in range(i + 1, len(dots)):
                arrow = DoubleArrow(dots[i].get_center(), dots[j].get_center())
                self.play(GrowArrow(arrow))

        border = [(-5, -3.5, 0),
                    (-5, 3.5, 0),
                    (5, 3.5, 0),
                    (5, -3.5, 0)]

        S_box = Polygon(*border, color=GREEN)
        S_label = TextMobject("$S_{\cup}$").scale(2).to_corner(corner=UP+RIGHT).shift(LEFT*0.25)

        self.play(ShowCreation(S_box), Write(S_label))

        self.wait(1)