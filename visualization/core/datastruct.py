import numpy as np
from dataclasses import dataclass, field
from typing import List

# Configuration and Data Structures
@dataclass
class Point:
    x: float
    y: float
    
    def distance_to(self, other: 'Point') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

@dataclass
class Machine:
    id: int
    shape: str  # 'rectangle' or 'l_shape'
    width: float
    height: float
    access_point: Point  # Relative to machine center
    position: Point = field(default_factory=lambda: Point(0, 0))
    rotation: float = 0  # In degrees
    l_cutout_width: float = 0  # For L-shaped machines
    l_cutout_height: float = 0
    
    def get_corners(self) -> List[Point]:
        """Get machine corners in world coordinates"""
        if self.shape == 'rectangle':
            corners = [
                Point(-self.width/2, -self.height/2),
                Point(self.width/2, -self.height/2),
                Point(self.width/2, self.height/2),
                Point(-self.width/2, self.height/2)
            ]
        else:  # L-shape
            corners = [
                Point(-self.width/2, -self.height/2),
                Point(self.width/2, -self.height/2),
                Point(self.width/2, self.height/2 - self.l_cutout_height),
                Point(self.width/2 - self.l_cutout_width, self.height/2 - self.l_cutout_height),
                Point(self.width/2 - self.l_cutout_width, self.height/2),
                Point(-self.width/2, self.height/2)
            ]
        
        # Apply rotation and translation
        rad = np.radians(self.rotation)
        cos_r, sin_r = np.cos(rad), np.sin(rad)
        
        world_corners = []
        for corner in corners:
            # Rotate
            x_rot = corner.x * cos_r - corner.y * sin_r
            y_rot = corner.x * sin_r + corner.y * cos_r
            # Translate
            world_corners.append(Point(
                x_rot + self.position.x,
                y_rot + self.position.y
            ))
        
        return world_corners
    
    def get_access_point_world(self) -> Point:
        """Get access point in world coordinates"""
        rad = np.radians(self.rotation)
        cos_r, sin_r = np.cos(rad), np.sin(rad)
        
        # Rotate access point
        x_rot = self.access_point.x * cos_r - self.access_point.y * sin_r
        y_rot = self.access_point.x * sin_r + self.access_point.y * cos_r
        
        return Point(
            x_rot + self.position.x,
            y_rot + self.position.y
        )