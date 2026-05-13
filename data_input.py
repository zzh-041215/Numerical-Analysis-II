from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


BASE_DIR = Path(__file__).resolve().parent
GLOBAL_PROPERTIES_CSV = BASE_DIR / "polytrope_global_properties.csv"
LANE_EMDEN_TABLE_CSV = BASE_DIR / "lane_emden_tables_no_page.csv"
MIN_N = 0.0
MAX_N_EXCLUSIVE = 5.0


@dataclass(frozen=True)
class GlobalProperty:
    n: float
    xi_1: Optional[float]
    theta_prime_surface: Optional[float]
    mass: Optional[float]
    rho_c_over_rho_avg: Optional[float]
    W_n: Optional[float]
    f_M: Optional[float]
    five_minus_n_f_A: Optional[float]
    x_eq: Optional[float]
    virial: Optional[float]


@dataclass(frozen=True)
class LaneEmdenPoint:
    geometry: str
    n: float
    xi: float
    theta: float
    theta_prime: Optional[float]
    theta_n_plus_1: Optional[float]


def _to_float(value: str) -> Optional[float]:
    value = value.strip()
    if value == "":
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    if math.isnan(result):
        return None
    return result


def _is_supported_n(n: float) -> bool:
    return MIN_N <= n < MAX_N_EXCLUSIVE


def load_global_properties(
    csv_path: Path | str = GLOBAL_PROPERTIES_CSV,
) -> Dict[float, GlobalProperty]:
    """Read polytrope_global_properties.csv and store rows by polytropic index n."""
    csv_path = Path(csv_path)
    properties: Dict[float, GlobalProperty] = {}

    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            n = _to_float(row["n"])
            if n is None:
                continue
            if not _is_supported_n(n):
                continue

            properties[n] = GlobalProperty(
                n=n,
                xi_1=_to_float(row.get("xi_1", "")),
                theta_prime_surface=_to_float(row.get("theta_prime_surface", "")),
                mass=_to_float(row.get("mass", "")),
                rho_c_over_rho_avg=_to_float(row.get("rho_c_over_rho_avg", "")),
                W_n=_to_float(row.get("W_n", "")),
                f_M=_to_float(row.get("f_M", "")),
                five_minus_n_f_A=_to_float(row.get("(5-n)f_A", "")),
                x_eq=_to_float(row.get("x_eq", "")),
                virial=_to_float(row.get("virial", "")),
            )

    return properties


def load_lane_emden_sphere_table(
    csv_path: Path | str = LANE_EMDEN_TABLE_CSV,
) -> Dict[float, List[LaneEmdenPoint]]:
    """Read Lane-Emden table data, keeping only geometry=Sphere and numeric rows."""
    csv_path = Path(csv_path)
    tables: Dict[float, List[LaneEmdenPoint]] = {}

    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if row.get("geometry", "").strip() != "Sphere":
                continue

            n = _to_float(row.get("n", ""))
            xi = _to_float(row.get("xi", ""))
            theta = _to_float(row.get("theta", ""))
            if n is None or xi is None or theta is None:
                continue
            if not _is_supported_n(n):
                continue
            if xi < 0:
                continue

            point = LaneEmdenPoint(
                geometry="Sphere",
                n=n,
                xi=xi,
                theta=theta,
                theta_prime=_to_float(row.get("theta_prime", "")),
                theta_n_plus_1=_to_float(row.get("theta_n_plus_1", "")),
            )
            tables.setdefault(n, []).append(point)

    for n, points in tables.items():
        tables[n] = sorted(points, key=lambda point: point.xi)

    return tables


class LaneEmdenReferenceData:
    """Convenience wrapper for special-point and global-function comparisons."""

    def __init__(
        self,
        global_properties: Dict[float, GlobalProperty],
        sphere_tables: Dict[float, List[LaneEmdenPoint]],
    ) -> None:
        self.global_properties = global_properties
        self.sphere_tables = sphere_tables

    @classmethod
    def from_csv(
        cls,
        global_csv: Path | str = GLOBAL_PROPERTIES_CSV,
        table_csv: Path | str = LANE_EMDEN_TABLE_CSV,
    ) -> "LaneEmdenReferenceData":
        return cls(
            global_properties=load_global_properties(global_csv),
            sphere_tables=load_lane_emden_sphere_table(table_csv),
        )

    def available_global_n(self) -> List[float]:
        return sorted(self.global_properties)

    def available_sphere_n(self) -> List[float]:
        return sorted(self.sphere_tables)

    def get_global_property(self, n: float) -> Optional[GlobalProperty]:
        return self.global_properties.get(float(n))

    def get_first_zero(self, n: float, *, allow_table_fallback: bool = False) -> Optional[float]:
        """Return xi_1 from the global properties table.

        Set allow_table_fallback=True only when the function table has been checked,
        because extracted PDF tables can contain malformed rows that create false
        sign changes.
        """
        n = float(n)
        prop = self.global_properties.get(n)
        if prop is not None and prop.xi_1 is not None and math.isfinite(prop.xi_1):
            return prop.xi_1

        if not allow_table_fallback:
            return None

        points = self.sphere_tables.get(n, [])
        for left, right in zip(points, points[1:]):
            if left.theta == 0:
                return left.xi
            if left.theta * right.theta < 0:
                ratio = abs(left.theta) / (abs(left.theta) + abs(right.theta))
                return left.xi + ratio * (right.xi - left.xi)

        return None

    def get_sphere_table(
        self,
        n: float,
        *,
        until_first_zero: bool = True,
    ) -> List[LaneEmdenPoint]:
        """Return Sphere table for n, optionally truncated to xi <= xi_1."""
        n = float(n)
        points = list(self.sphere_tables.get(n, []))
        if not until_first_zero:
            return points

        xi_1 = self.get_first_zero(n)
        if xi_1 is None:
            return points
        return [point for point in points if point.xi <= xi_1]

    def interpolate_theta(self, n: float, xi: float) -> Optional[float]:
        """Linear interpolation of theta from the Sphere reference table."""
        points = self.get_sphere_table(n, until_first_zero=True)
        if not points or xi < points[0].xi or xi > points[-1].xi:
            return None

        for left, right in zip(points, points[1:]):
            if left.xi <= xi <= right.xi:
                if right.xi == left.xi:
                    return left.theta
                weight = (xi - left.xi) / (right.xi - left.xi)
                return left.theta + weight * (right.theta - left.theta)

        if xi == points[-1].xi:
            return points[-1].theta
        return None


def load_reference_data() -> LaneEmdenReferenceData:
    return LaneEmdenReferenceData.from_csv()


def _format_list(values: Iterable[float]) -> str:
    return ", ".join(f"{value:g}" for value in values)


if __name__ == "__main__":
    data = load_reference_data()
    print("Global-property n values:")
    print(_format_list(data.available_global_n()))
    print()
    print("Sphere table n values:")
    print(_format_list(data.available_sphere_n()))
    print()

    for n in data.available_sphere_n():
        points = data.get_sphere_table(n, until_first_zero=True)
        xi_1 = data.get_first_zero(n)
        print(f"n={n:g}: {len(points)} Sphere points before xi_1, xi_1={xi_1}")
