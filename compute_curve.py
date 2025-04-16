import math
import numpy as np
import matplotlib.pyplot as plt

class curved_tool:
    """
    Class to represent a curved tool.
    The curve is defined by segments of type 'line' or 'arc'.
    The 'line' segment is ((x1, y1), (x2, y2)).
    The 'arc' segment is (center, radius, start_angle, end_angle).
    """
    def __init__(self, curve_parameters):
        """
        Initialize the curved tool with the curve parameters.
        curve_parameters: list of tuples, each describing a segment, e.g.:
          [
            ('line', (0,0), (1,0)),
            ('arc', (1,1), 1.0, 0, math.pi/2),
            ...
          ]
        """
        self.curve_parameters = curve_parameters

    def get_curve_parameters(self):
        return self.curve_parameters
    
    @staticmethod
    def draw_line(ax, p1, p2, color='black'):
        """
        Draw a line between two points on a matplotlib Axes.
        """
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color)
        return ax

    @staticmethod
    def draw_arc(ax, center, radius, start_angle, end_angle, color='black'):
        """
        Draw an arc on a matplotlib Axes.
        center: (cx, cy)
        radius: float
        start_angle, end_angle: in radians
        """
        # Discretize the arc
        arc_thetas = np.linspace(start_angle, end_angle, 100)
        x = center[0] + radius * np.cos(arc_thetas)
        y = center[1] + radius * np.sin(arc_thetas)
        ax.plot(x, y, color=color)
        return ax

    def draw_curve(self, ax=None, color='black'):
        """
        Draws the entire curve on the given matplotlib Axes (or a new figure if ax=None).
        """
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
        Given a list of angles, compute the minimal bounding interval [angle_min, angle_max].
        Return (cone_width, cone_axis).
        - cone_width: difference angle_max - angle_min
        - cone_axis: midpoint angle (normalized to [-pi, pi])
        """
        if not angles.any():
            return 0.0, 0.0
        
        # Sort angles, unwrap around 2*pi to handle wraparound
        angles_sorted = np.sort(np.unwrap(angles))
        # The largest gap is the region NOT visible. The complement is the minimal bounding interval.
        # We'll do an approach similar to the one used for circular intervals:
        angles_extended = np.concatenate([angles_sorted, [angles_sorted[0] + 2*np.pi]])
        diffs = np.diff(angles_extended)
        gap_idx = np.argmax(diffs)
        largest_gap = diffs[gap_idx]
        cone_width = 2*np.pi - largest_gap
        visible_start = angles_extended[gap_idx+1] % (2*np.pi)
        axis = visible_start + cone_width/2
        axis = (axis + np.pi) % (2*np.pi) - np.pi  # normalize
        return cone_width, axis

    @staticmethod
    def cone_to_line(line_parameters, query):
        """
        Compute the visibility cone from a single query (or multiple queries) to a line segment.
        line_parameters = (p1, p2), each p is (x, y)
        query: (qx, qy) or array of shape (N,2)
        
        Returns: (cone_width, cone_axis)
        """
        p1, p2 = line_parameters
        # If multiple queries, handle them one by one. For a single query, convert to array for uniformity.
        query = np.array(query, ndmin=2)
        
        # Sample the line segment
        # A small sampling to approximate the line
        num_samples = 50
        t = np.linspace(0, 1, num_samples)
        x_line = p1[0] + t*(p2[0] - p1[0])
        y_line = p1[1] + t*(p2[1] - p1[1])
        line_points = np.stack([x_line, y_line], axis=-1)  # shape (50, 2)
        
        # For each query, gather angles
        # We'll do it for the FIRST query if there's more than one.
        q = query[0]
        vectors = line_points - q
        angles = np.arctan2(vectors[:,1], vectors[:,0])
        
        return curved_tool._bounding_cone(angles)

    @staticmethod
    def cone_to_arc(arc_parameters, query):
        """
        Compute the visibility cone from a query point(s) to an arc segment.
        arc_parameters = (center, radius, start_angle, end_angle)
        query: (qx, qy) or array of shape (N,2)
        
        Returns: (cone_width, cone_axis)
        """
        center, radius, start_angle, end_angle = arc_parameters
        query = np.array(query, ndmin=2)
        
        # Sample the arc
        num_samples = 50
        arc_thetas = np.linspace(start_angle, end_angle, num_samples)
        x_arc = center[0] + radius * np.cos(arc_thetas)
        y_arc = center[1] + radius * np.sin(arc_thetas)
        arc_points = np.stack([x_arc, y_arc], axis=-1)
        
        # Use first query if multiple
        q = query[0]
        vectors = arc_points - q
        angles = np.arctan2(vectors[:,1], vectors[:,0])
        
        return curved_tool._bounding_cone(angles)

    def cone_to_curve(self, query):
        """
        Compute the visibility cone from a query point to the entire curve
        by combining the cone intervals from each segment.
        
        Returns: (cone_width, cone_axis)
        """
        all_angles = []
        # For each segment, sample it and gather angles
        for seg in self.curve_parameters:
            seg_type = seg[0]
            if seg_type == 'line':
                _, p1, p2 = seg
                width, axis = self.cone_to_line((p1, p2), query)
                # We'll reconstruct angles from width + axis
                # We'll store them in an offset to unify them later
                # But simpler is to sample directly:
                # => we already have them above
                # Instead let's do: reconstruct angles from the bounding interval
                # But we want the actual angles from the segment for merging
                pass  # We'll do a direct approach below
            elif seg_type == 'arc':
                _, center, radius, start_angle, end_angle = seg
                pass
        
        # Simpler approach: directly sample entire curve at once
        # (We assume continuity isn't an issue.)
        big_points = []
        for seg in self.curve_parameters:
            if seg[0] == 'line':
                _, p1, p2 = seg
                num_samples = 50
                t = np.linspace(0,1,num_samples)
                line_pts = np.stack([p1[0]+t*(p2[0]-p1[0]),
                                     p1[1]+t*(p2[1]-p1[1])], axis=-1)
                big_points.append(line_pts)
            elif seg[0] == 'arc':
                _, center, radius, start_angle, end_angle = seg
                num_samples = 50
                arc_thetas = np.linspace(start_angle, end_angle, num_samples)
                x_arc = center[0] + radius*np.cos(arc_thetas)
                y_arc = center[1] + radius*np.sin(arc_thetas)
                arc_pts = np.stack([x_arc, y_arc], axis=-1)
                big_points.append(arc_pts)
        
        big_points = np.concatenate(big_points, axis=0) if big_points else np.zeros((0,2))
        
        # Compute angles from query to all big_points
        q = np.array(query)
        vectors = big_points - q
        angles = np.arctan2(vectors[:,1], vectors[:,0])
        
        return self._bounding_cone(angles)

    def grid_metric(self, xlim, ylim, resolution=50):
        """
        Evaluate cone_to_curve(...) over a grid of query points in [xlim, ylim].
        Returns (grid_x, grid_y, cone_widths, axis_x, axis_y).
        """
        x_vals = np.linspace(xlim[0], xlim[1], resolution)
        y_vals = np.linspace(ylim[0], ylim[1], resolution)
        grid_x, grid_y = np.meshgrid(x_vals, y_vals)
        
        cone_widths = np.zeros_like(grid_x)
        axis_x = np.zeros_like(grid_x)
        axis_y = np.zeros_like(grid_x)
        
        for i in range(resolution):
            for j in range(resolution):
                q = (grid_x[i,j], grid_y[i,j])
                width, axis = self.cone_to_curve(q)
                cone_widths[i,j] = width
                axis_x[i,j] = np.cos(axis)
                axis_y[i,j] = np.sin(axis)
        
        return grid_x, grid_y, cone_widths, axis_x, axis_y
    
    def plot_heatmap(self, grid_x, grid_y, cone_widths, axis_x, axis_y):
        """
        Plot the cone width as a heatmap, with small arrows for the axis direction.
        """
        fig, ax = plt.subplots(figsize=(8,6))
        im = ax.pcolormesh(grid_x, grid_y, cone_widths, shading='auto', cmap='plasma')
        cb = plt.colorbar(im, ax=ax, label='Visibility Cone Width (radians)')
        # Plot small quiver for directions
        # sample every 5th point for quiver
        step = max(1, cone_widths.shape[0] // 20)
        axis_x = axis_x[::step, ::step]
        axis_y = axis_y[::step, ::step]
        grid_x = grid_x[::step, ::step]
        grid_y = grid_y[::step, ::step]
        # Normalize quiver length
        norm = np.sqrt(axis_x**2 + axis_y**2)
        axis_x /= norm*10
        axis_y /= norm*10
        ax.quiver(grid_x, grid_y, axis_x, axis_y, color='black', scale=0.1)
        
        ax.set_aspect('equal', 'box')
        ax.set_title("Cone Width Heatmap + Axis Directions")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        return ax

    def grid_metric_modified(self, xlim, ylim, resolution=50):
        """
        Placeholder for a version that considers some curve segments as obstacles.
        (Not implemented here—depends on your application logic.)
        """
        pass

# ===================== Example Usage =====================
if __name__ == "__main__":
    # Example curve: one line from (0,0) to (2,0),
    # then an arc from center=(2,1) radius=1 from -pi/2 to pi/2
    curve_segments = [
        ("line", (0,0), (2,0)),
        ("arc", (2,1), 1.0, -math.pi/2, math.pi/2)
    ]
    tool = curved_tool(curve_segments)
    
    # Evaluate the grid metric
    grid_x, grid_y, cone_widths, ax_x, ax_y = tool.grid_metric(xlim=(-1,4), ylim=(-1,3), resolution=100)
    
    # Plot the results
    fig_ax = tool.plot_heatmap(grid_x, grid_y, cone_widths, ax_x, ax_y)
    # Optionally draw the curve on top
    tool.draw_curve(fig_ax)
    plt.show()