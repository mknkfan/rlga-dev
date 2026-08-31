from core.datastruct import Point
from typing import List, Tuple, Optional, Dict

class CollisionDetector:
    @staticmethod
    def point_in_polygon(point: Point, polygon: List[Point]) -> bool:
        """Check if point is inside polygon using ray casting"""
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0].x, polygon[0].y
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n].x, polygon[i % n].y
            if point.y > min(p1y, p2y):
                if point.y <= max(p1y, p2y):
                    if point.x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (point.y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or point.x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    @staticmethod
    def polygons_intersect(poly1: List[Point], poly2: List[Point]) -> bool:
        """Check if two polygons intersect"""
        # Check if any vertex of poly1 is inside poly2
        for point in poly1:
            if CollisionDetector.point_in_polygon(point, poly2):
                return True
        
        # Check if any vertex of poly2 is inside poly1
        for point in poly2:
            if CollisionDetector.point_in_polygon(point, poly1):
                return True
        
        # Check edge intersections
        for i in range(len(poly1)):
            for j in range(len(poly2)):
                if CollisionDetector.line_segments_intersect(
                    poly1[i], poly1[(i+1)%len(poly1)],
                    poly2[j], poly2[(j+1)%len(poly2)]
                ):
                    return True
        
        return False
    
    @staticmethod
    def line_segments_intersect(p1: Point, q1: Point, p2: Point, q2: Point) -> bool:
        """Check if two line segments intersect"""
        def orientation(p: Point, q: Point, r: Point) -> int:
            val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
            if val == 0:
                return 0  # collinear
            return 1 if val > 0 else 2  # clockwise or counterclockwise
        
        def on_segment(p: Point, q: Point, r: Point) -> bool:
            return (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and
                    q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y))
        
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)
        
        # General case
        if o1 != o2 and o3 != o4:
            return True
        
        # Special cases
        if (o1 == 0 and on_segment(p1, p2, q1)) or \
           (o2 == 0 and on_segment(p1, q2, q1)) or \
           (o3 == 0 and on_segment(p2, p1, q2)) or \
           (o4 == 0 and on_segment(p2, q1, q2)):
            return True
        
        return False