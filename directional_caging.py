import math
import numpy as np
import matplotlib.pyplot as plt

class curved_tool:
    """
    Class to represent a curved tool.
    The curve is constructed of a sequence of line and arc segments.
    Line segments are represented by their start and end points.
    Arc segments are represented by center, radius, start_angle and end_angle.
    Methods include cone_to_line, cone_to_arc, and cone_to_curve, which compute
    the visibility cone (here represented as a tuple of (width, axis)) from a query point.
    Additional methods compute metrics along the curve and plot the curve colored by a metric.
    """
    def __init__(self, curve_parameters):
        """
        Initialize the curved tool with the curve parameters.
        curve_parameters: list of tuples for the segments. For example:
           [('line', (0,0), (2,0)),
            ('arc', (2,1), 1.0, -math.pi/2, math.pi/2)]
        """
        self.curve_parameters = curve_parameters
        self._discretize_curve()

    def _discretize_curve(self, num_samples=500):
        big_points = []
        for seg in self.curve_parameters:
            if seg[0] == 'line':
                _, p1, p2 = seg
                t = np.linspace(0, 1, num_samples)
                pts = np.stack([p1[0] + t*(p2[0]-p1[0]),
                                p1[1] + t*(p2[1]-p1[1])], axis=-1)
                big_points.append(pts)
            elif seg[0] == 'arc':
                _, center, radius, start_angle, end_angle = seg
                arc_thetas = np.linspace(start_angle, end_angle, num_samples)
                pts = np.stack([center[0] + radius * np.cos(arc_thetas),
                                center[1] + radius * np.sin(arc_thetas)], axis=-1)
                big_points.append(pts)
        if big_points:
            big_points = np.concatenate(big_points, axis=0)
        else:
            big_points = np.zeros((0,2))
        self.curve_discretization = big_points

    def get_curve_parameters(self):
        return self.curve_parameters
    
    @staticmethod
    def draw_line(ax, p1, p2, color='black'):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color)
        return ax

    @staticmethod
    def draw_arc(ax, center, radius, start_angle, end_angle, color='black'):
        theta = np.linspace(start_angle, end_angle, 100)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        ax.plot(x, y, color=color)
        return ax

    def draw_curve(self, ax=None, color='black'):
        if ax is None:
            fig, ax = plt.subplots()
        for seg in self.curve_parameters:
            seg_type = seg[0]
            if seg_type == 'line':
                _, p1, p2 = seg
                self.draw_line(ax, p1, p2, color=color)
            elif seg_type == 'arc':
                _, center, radius, start_angle, end_angle = seg
                self.draw_arc(ax, center, radius, start_angle, end_angle, color=color)
        return ax

    @staticmethod
    def _bounding_cone(angles):
        """
        Given an array of angles (in radians), compute the minimal bounding interval.
        Returns (cone_width, cone_axis), where cone_axis is the midpoint angle.
        """
        if angles.size == 0:
            return 0.0, 0.0
        # Sort and unwrap angles to handle wrap-around
        angles_sorted = np.sort(np.unwrap(angles))
        # Append the first angle + 2*pi to account for wrap-around
        angles_extended = np.concatenate([angles_sorted, [angles_sorted[0] + 2*np.pi]])
        diffs = np.diff(angles_extended)
        gap_idx = np.argmax(diffs)
        largest_gap = diffs[gap_idx]
        cone_width = 2*np.pi - largest_gap
        visible_start = angles_extended[gap_idx + 1] % (2*np.pi)
        axis = visible_start + cone_width/2
        axis = (axis + np.pi) % (2*np.pi) - np.pi  # normalize to [-pi,pi]
        return cone_width, axis

    @staticmethod
    def cone_to_line(line_parameters, query):
        """
        Compute the visibility cone from a query point to a line segment.
        line_parameters: (p1, p2), where each p is (x, y)
        query: (qx, qy)
        Returns: (cone_width, cone_axis)
        """
        p1, p2 = line_parameters
        num_samples = 50
        t = np.linspace(0, 1, num_samples)
        x_line = p1[0] + t * (p2[0] - p1[0])
        y_line = p1[1] + t * (p2[1] - p1[1])
        points = np.stack([x_line, y_line], axis=-1)  # shape (num_samples, 2)
        vecs = points - np.array(query)
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        return curved_tool._bounding_cone(angles)

    @staticmethod
    def cone_to_arc(arc_parameters, query):
        """
        Compute the visibility cone from a query point to an arc.
        arc_parameters: (center, radius, start_angle, end_angle)
        query: (qx, qy)
        Returns: (cone_width, cone_axis)
        """
        center, radius, start_angle, end_angle = arc_parameters
        num_samples = 50
        arc_thetas = np.linspace(start_angle, end_angle, num_samples)
        x_arc = center[0] + radius * np.cos(arc_thetas)
        y_arc = center[1] + radius * np.sin(arc_thetas)
        points = np.stack([x_arc, y_arc], axis=-1)
        vecs = points - np.array(query)
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        return curved_tool._bounding_cone(angles)

    def cone_to_curve_old(self, query):
        """
        Compute the visibility cone from a query point to the entire curve.
        The curve is comprised of multiple segments (line and arc).
        Returns: (cone_width, cone_axis)
        """
        big_points = self.curve_discretization
        vecs = big_points - np.array(query)
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        return self._bounding_cone(angles)
    
    def cone_to_curve(self,query):
        big_points = self.curve_discretization
        # Compute angles from query to all sampled points
        vecs = big_points - np.array(query)
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        
        # EXTRA STEP: If the query is very close to the curve, identify the local neighborhood
        # and, if there is an overly large gap between the neighbors, add filler angles.
        dists = np.linalg.norm(big_points - np.array(query), axis=1)
        closest_idx = np.argmin(dists)
        if closest_idx > 0 and closest_idx < len(big_points) - 1:
            angle_left = np.arctan2(
                (big_points[closest_idx - 1] - np.array(query))[1],
                (big_points[closest_idx - 1] - np.array(query))[0]
            )
            angle_right = np.arctan2(
                (big_points[closest_idx + 1] - np.array(query))[1],
                (big_points[closest_idx + 1] - np.array(query))[0]
            )
            
            # Function for circular difference (returns value in [-pi,pi])
            def circ_diff(a, b):
                return (a - b + np.pi) % (2 * np.pi) - np.pi
            
            gap = abs(circ_diff(angle_right, angle_left))
            threshold_gap = np.pi * 0.8  # Adjust this threshold as needed
            
            if gap > threshold_gap:
                # Insert an extra angle: the midpoint of the gap
                mid_angles = np.linspace(angle_left, angle_right, num=100)
                angles = np.concatenate([angles, mid_angles])
        
        # Compute the minimal bounding cone from the final set of angles.
        return self._bounding_cone(angles)

    def grid_metric(self, xlim, ylim, resolution=200):
        """
        Evaluate cone_to_curve at a grid of query points over the rectangle defined by xlim and ylim.
        Returns: (grid_x, grid_y, cone_widths, axis_x, axis_y)
        where axis_x and axis_y are the cosine and sine of the cone axis.
        """
        x_vals = np.linspace(xlim[0], xlim[1], resolution)
        y_vals = np.linspace(ylim[0], ylim[1], resolution)
        grid_x, grid_y = np.meshgrid(x_vals, y_vals)
        cone_widths = np.zeros_like(grid_x)
        axis_x = np.zeros_like(grid_x)
        axis_y = np.zeros_like(grid_x)
        for i in range(resolution):
            for j in range(resolution):
                q = (grid_x[i, j], grid_y[i, j])
                width, axis = self.cone_to_curve(q)
                cone_widths[i, j] = width
                axis_x[i, j] = np.cos(axis)
                axis_y[i, j] = np.sin(axis)
        return grid_x, grid_y, cone_widths, axis_x, axis_y

    def plot_heatmap(self, grid_x, grid_y, cone_widths, axis_x, axis_y):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(grid_x, grid_y, cone_widths, shading='auto', cmap='plasma')
        plt.colorbar(im, ax=ax, label='Visibility Cone Width (radians)')
        # sample every 5th point for quiver
        step = max(1, cone_widths.shape[0] // 20)
        axis_x = axis_x[::step, ::step]
        axis_y = axis_y[::step, ::step]
        grid_x = grid_x[::step, ::step]
        grid_y = grid_y[::step, ::step]
        # Normalize quiver length
        norm = np.sqrt(axis_x**2 + axis_y**2)
        axis_x /= norm*500
        axis_y /= norm*500
        ax.quiver(grid_x, grid_y, axis_x, axis_y, color='white', scale=0.1, width=0.002)
        ax.set_aspect('equal', 'box')
        ax.set_title("Cone Width Heatmap + Axis Directions")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        return ax

    def compute_curve_metrics_old(self, step=0.01):
        """
        Iterates over each segment of the curve, discretizes the segment (parameter t from 0 to 1
        with the given step size), computes the visibility cone (using cone_to_curve with query at the point),
        and returns a list of tuples: (point, cone_width). The cone width is taken as the "metric".
        """
        metrics = []  # list of tuples: (point, cone_width)
        eps = 1e-4  # small value to avoid the exact curve
        for seg in self.curve_parameters:
            if seg[0] == 'line':
                _, p1, p2 = seg
                t_vals = np.arange(0, 1 + step, step)
                for t in t_vals:
                    x = p1[0] + t * (p2[0] - p1[0])
                    y = p1[1] + t * (p2[1] - p1[1])
                    point = (x, y)
                    width, _ = self.cone_to_curve(point)
                    metrics.append((point, width))
            elif seg[0] == 'arc':
                _, center, radius, start_angle, end_angle = seg
                t_vals = np.arange(0, 1 + step, step)
                for t in t_vals:
                    theta = start_angle + t * (end_angle - start_angle)
                    x = center[0] + radius*(1-eps) * np.cos(theta)
                    y = center[1] + radius *(1-eps) * np.sin(theta)
                    point = (x, y)
                    width, _ = self.cone_to_curve(point)
                    metrics.append((point, width))
        return metrics
    
    def compute_curve_outline(self, step=0.01, eps=0.2):
        """
        Discretizes the entire curve and computes an outline surrounding it at a distance 'eps'.
        The outline is defined as the union of the left offset and the right offset, forming
        a closed loop.
        
        Parameters:
            step (float): Discretization step in parameter space for each segment.
            eps (float): Offset distance from the curve.
            
        Returns:
            outline (numpy.ndarray): Array of shape (N,2) of points forming the closed outline.
        """
        # Discretize the curve: accumulate points from all segments.
        curve_points = []
        for seg in self.curve_parameters:
            if seg[0] == 'line':
                # seg: ('line', p1, p2)
                _, p1, p2 = seg
                t_vals = np.arange(0, 1 + step, step)
                for t in t_vals:
                    x = p1[0] + t * (p2[0] - p1[0])
                    y = p1[1] + t * (p2[1] - p1[1])
                    curve_points.append((x, y))
            elif seg[0] == 'arc':
                # seg: ('arc', center, radius, start_angle, end_angle)
                _, center, radius, start_angle, end_angle = seg
                t_vals = np.arange(0, 1 + step, step)
                for t in t_vals:
                    theta = start_angle + t * (end_angle - start_angle)
                    # Slightly shrink the arc (if desired) with a multiplicative factor (optional)
                    x = center[0] + radius * np.cos(theta)
                    y = center[1] + radius * np.sin(theta)
                    curve_points.append((x, y))
                    
        curve_points = np.array(curve_points)
        n_points = len(curve_points)
        
        # Compute left and right offsets using approximated tangents.
        left_offsets = []
        right_offsets = []
        for i, pt in enumerate(curve_points):
            if i == 0:
                tangent = curve_points[i+1] - pt
            elif i == n_points - 1:
                tangent = pt - curve_points[i-1]
            else:
                tangent = curve_points[i+1] - curve_points[i-1]
            t_norm = np.linalg.norm(tangent)
            if t_norm == 0:
                tangent = np.array([1, 0])
            else:
                tangent = tangent / t_norm
            
            # Left normal is defined as (-tangent_y, tangent_x)
            normal = np.array([-tangent[1], tangent[0]])
            left_offsets.append(pt + eps * normal)
            right_offsets.append(pt - eps * normal)
            
        left_offsets = np.array(left_offsets)
        right_offsets = np.array(right_offsets)
        
        # Form a closed outline: left offset in order, then right offset in reverse.
        outline = np.concatenate([left_offsets, right_offsets[::-1]], axis=0)
        return outline

    def compute_curve_metrics(self, step=0.01):

        outline = self.compute_curve_outline(step=step, eps=2.5)
        metrics = []  # list of tuples: (point, cone_width)
        for i in range(len(outline)):
            point = outline[i]
            width, _ = self.cone_to_curve(point)
            metrics.append((point, width))
        return metrics


    def draw_curve_colormap(self, step=0.01, cmap='viridis'):
        """
        Plots the curve with points colored according to the computed cone metric (cone width).
        Returns a matplotlib Axes with the plot.
        """
        metrics = self.compute_curve_metrics(step=step)
        points = np.array([pt for pt, _ in metrics])
        values = np.array([val for _, val in metrics])
        fig, ax = plt.subplots(figsize=(8,6))
        sc = ax.scatter(points[:, 0], points[:, 1], c=values, cmap=cmap, s=20, edgecolor='none')
        plt.colorbar(sc, ax=ax, label='Visibility Cone Width (radians)')
        ax.set_aspect('equal', 'box')
        ax.set_title("Curve Colored by Visibility Cone Width")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        # Also draw the underlying curve in a light gray line for reference.
        # self.draw_curve(ax, color='lightgray')
        plt.show()
        return ax

# ===================== Example Usage =====================
if __name__ == "__main__":
    # first accessible parametrization for the segments, according to the design sketch
    ls1 = 25
    arcs1 = [60.75,2*math.pi/9] #
    arcs2 = [5.5,math.pi*7/9]   
    ls2 = 40
    # this is not a completely general formula for any curve, but works for the line-arc-arc-line structure
    centerarc1 = np.array([arcs1[0], ls1])
    anglesarc1 = np.array([math.pi, math.pi-arcs1[1]])
    centerarc2 = centerarc1 +  (arcs1[0]+arcs2[0])*np.array([math.cos(anglesarc1[1]), math.sin(anglesarc1[1])]) # when two arcs are tangent, a line of (r1+r2) connects their centers
    # the second arc starts from the negative to the angle the first one finishes at
    anglesarc2 = [-arcs1[1], -arcs1[1]+arcs2[1]]
    ls2start = centerarc2 + arcs2[0]*np.array([math.cos(anglesarc2[1]), math.sin(anglesarc2[1])])
    ls2end = ls2start + ls2 * np.array([math.sin(-anglesarc2[1]), math.cos(anglesarc2[1])])

    curve_segments = [
        ("line", (0, 0), (0, ls1)),
        ("arc", centerarc1, arcs1[0], anglesarc1[0], anglesarc1[1]),
        ("arc", centerarc2, arcs2[0], anglesarc2[0], anglesarc2[1]),
        ("line", ls2start, ls2end)
    ]
    tool = curved_tool(curve_segments)

    # Example: Compute grid metric and plot heatmap over a specified region.
    grid_x, grid_y, cone_widths, ax_x, ax_y = tool.grid_metric(xlim=(-50, 50), ylim=(-10, 100), resolution=50)
    ax = tool.plot_heatmap(grid_x, grid_y, cone_widths, ax_x, ax_y)
    tool.draw_curve(ax, color='white')
    plt.show()

    # Example: Compute per-point metrics along the curve and visualize with a colormap.
    tool.draw_curve_colormap(step=0.01, cmap='viridis')
