"""
Restricted Area & Geofencing Intrusion Detection Module.
"""
from typing import List, Dict, Any, Tuple, Optional
from shapely.geometry import Point, Polygon


class RestrictedZone:
    def __init__(self, zone_id: str, name: str, coordinates: List[Tuple[float, float]], 
                 severity: str = "HIGH", zone_type: str = "RESTRICTED", color: str = "#DC2626"):
        self.zone_id = zone_id
        self.name = name
        self.coordinates = coordinates
        self.severity = severity
        self.zone_type = zone_type
        self.color = color
        self.polygon = Polygon([(lon, lat) for lat, lon in coordinates])

    def contains_point(self, lat: float, lon: float) -> bool:
        pt = Point(lon, lat)
        return self.polygon.contains(pt) or self.polygon.touches(pt)


class GeofenceManager:
    def __init__(self):
        self.zones: Dict[str, RestrictedZone] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self._load_default_zones()

    def _load_default_zones(self):
        zone1_coords = [
            (37.7760, -122.4210), (37.7780, -122.4210), (37.7780, -122.4180),
            (37.7760, -122.4180), (37.7760, -122.4210)
        ]
        self.add_zone(RestrictedZone("ZONE-RED-01", "Red Zone: Research Compound", zone1_coords, "CRITICAL", "NO_ENTRY", "#DC2626"))

        zone2_coords = [
            (37.7735, -122.4225), (37.7750, -122.4225), (37.7750, -122.4200),
            (37.7735, -122.4200), (37.7735, -122.4225)
        ]
        self.add_zone(RestrictedZone("ZONE-AMBER-02", "Amber Zone: Helipad & Fuel", zone2_coords, "WARNING", "AUTHORIZED_ONLY", "#D97706"))

    def add_zone(self, zone: RestrictedZone):
        self.zones[zone.zone_id] = zone

    def remove_zone(self, zone_id: str):
        if zone_id in self.zones:
            del self.zones[zone_id]

    def get_all_zones(self) -> List[RestrictedZone]:
        return list(self.zones.values())

    def check_intrusion(self, track_id: int, class_name: str, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        for zone in self.zones.values():
            if zone.contains_point(lat, lon):
                alert = {
                    "zone_id": zone.zone_id, "zone_name": zone.name, "severity": zone.severity,
                    "track_id": track_id, "class_name": class_name, "lat": lat, "lon": lon,
                    "description": f"Unauthorized {class_name.upper()} (ID: #{track_id}) intruded into {zone.name}."
                }
                self.alert_history.append(alert)
                return alert
        return None
