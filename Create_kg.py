"""Milestone 2 Neo4j loader.

The loader is built in stages: entity nodes and race relationships are loaded
before the result, qualifying, and pit-stop fact nodes.
"""

from __future__ import annotations

import csv
import os
from datetime import date
from pathlib import Path
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parent
CSV_DIR = Path(os.environ.get("F1_CSV_DIR", ROOT_DIR / "cleaned_f1"))
CONFIG_PATH = Path(os.environ.get("F1_NEO4J_CONFIG", ROOT_DIR / "config.txt"))


def read_config(path: Path = CONFIG_PATH) -> dict[str, str]:
    """Read Neo4j settings by key from a KEY=VALUE configuration file."""
    settings: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid config line {line_number}: expected KEY=VALUE")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key:
            raise ValueError(f"Invalid config line {line_number}: empty key")
        settings[key] = value

    required = {"URI", "USERNAME", "PASSWORD"}
    missing = sorted(required - settings.keys())
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")
    return settings


def neo4j_driver(settings: dict[str, str]):
    """Create a Neo4j driver using the configured URI and credentials."""
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError(
            "The neo4j package is required. Install it with: pip install neo4j"
        ) from exc

    uri = settings["URI"]
    secure_scheme = uri.startswith(("neo4j+s://", "bolt+s://"))
    if secure_scheme:
        uri = uri.replace("+s://", "://", 1)
        try:
            import certifi
            import ssl
        except ImportError as exc:
            raise RuntimeError(
                "The certifi package is required for secure Neo4j connections. "
                "Install it with: pip install certifi"
            ) from exc
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        return GraphDatabase.driver(
            uri,
            auth=(settings["USERNAME"], settings["PASSWORD"]),
            ssl_context=ssl_context,
        )

    return GraphDatabase.driver(
        uri,
        auth=(settings["USERNAME"], settings["PASSWORD"]),
    )


def csv_rows(table_name: str, csv_dir: Path = CSV_DIR) -> Iterable[dict[str, str]]:
    """Read rows from one of the Milestone 2 CSV files without changing the source."""
    path = csv_dir / f"{table_name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV file was not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as source:
        yield from csv.DictReader(source)


def omit_missing(properties: dict[str, Any]) -> dict[str, Any]:
    """Remove CSV missing values so Neo4j does not receive sentinel properties."""
    return {
        key: value
        for key, value in properties.items()
        if value is not None and value != "" and value != r"\N"
    }


def integer(value: str | None) -> int | None:
    if value is None or value.strip() in ("", r"\N") or value.strip().lower() == "nan":
        return None
    numeric = float(value)
    if not numeric.is_integer():
        raise ValueError(f"Expected a whole-number value, got {value!r}")
    return int(numeric)


def decimal(value: str | None) -> float | None:
    if value is None or value.strip() in ("", r"\N") or value.strip().lower() == "nan":
        return None
    return float(value)


def neo4j_date(value: str | None) -> date | None:
    if value is None or value.strip() in ("", r"\N") or value.strip().lower() in (
        "",
        "nan",
        "nat",
    ):
        return None
    return date.fromisoformat(value)


ENTITY_CONSTRAINTS = (
    ("Driver", "driverId"),
    ("Constructor", "constructorId"),
    ("Circuit", "circuitId"),
    ("Season", "year"),
    ("Status", "statusId"),
    ("Race", "raceId"),
    ("Result", "resultId"),
    ("Qualifying", "qualifyId"),
    ("PitStop", "eventId"),
)


def create_constraints(tx: Any) -> None:
    """Create idempotent uniqueness constraints for every schema node type."""
    for label, property_name in ENTITY_CONSTRAINTS:
        constraint_name = f"{label.lower()}_{property_name}_unique"
        tx.run(
            f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS "
            f"FOR (node:{label}) REQUIRE node.{property_name} IS UNIQUE"
        )


def load_entity_batch(tx: Any, label: str, key: str, rows: list[dict[str, Any]]) -> None:
    """Merge one entity batch using its schema identifier."""
    tx.run(
        f"UNWIND $rows AS row "
        f"MERGE (node:{label} {{{key}: row.{key}}}) "
        "SET node += row",
        rows=rows,
    )


def load_race_relationships(tx: Any, rows: list[dict[str, int]]) -> None:
    """Merge the required Race-to-Circuit and Race-to-Season relationships."""
    tx.run(
        "UNWIND $rows AS row "
        "MATCH (race:Race {raceId: row.raceId}) "
        "MATCH (circuit:Circuit {circuitId: row.circuitId}) "
        "MERGE (race)-[:AT_CIRCUIT]->(circuit)",
        rows=rows,
    )
    tx.run(
        "UNWIND $rows AS row "
        "MATCH (race:Race {raceId: row.raceId}) "
        "MATCH (season:Season {year: row.year}) "
        "MERGE (race)-[:IN_SEASON]->(season)",
        rows=rows,
    )


def entity_rows(csv_name: str) -> list[dict[str, Any]]:
    """Build typed properties for one parent node table."""
    rows: list[dict[str, Any]] = []
    for source in csv_rows(csv_name):
        if csv_name == "drivers":
            properties = {
                "driverId": integer(source["driverId"]),
                "driverRef": source["driverRef"],
                "number": integer(source["number"]),
                "code": source["code"],
                "forename": source["forename"],
                "surname": source["surname"],
                "dob": neo4j_date(source["dob"]),
                "nationality": source["nationality"],
            }
        elif csv_name == "constructors":
            properties = {
                "constructorId": integer(source["constructorId"]),
                "constructorRef": source["constructorRef"],
                "constructor_name": source["name"],
                "constructor_nationality": source["nationality"],
            }
        elif csv_name == "circuits":
            properties = {
                "circuitId": integer(source["circuitId"]),
                "circuitRef": source["circuitRef"],
                "circuit_name": source["name"],
                "circuit_loc": source["location"],
                "circuit_country": source["country"],
                "lat": decimal(source["lat"]),
                "lng": decimal(source["lng"]),
                "alt": integer(source["alt"]),
            }
        elif csv_name == "seasons":
            properties = {"year": integer(source["year"])}
        elif csv_name == "status":
            properties = {
                "statusId": integer(source["statusId"]),
                "status": source["status"],
            }
        elif csv_name == "races":
            properties = {
                "raceId": integer(source["raceId"]),
                "year": integer(source["year"]),
                "round": integer(source["round"]),
                "name": source["name"],
                "date": neo4j_date(source["date"]),
                "time": source["time"],
            }
        else:
            raise ValueError(f"Unsupported entity table: {csv_name}")
        rows.append(omit_missing(properties))
    return rows


def load_entities(session: Any) -> None:
    """Load all parent nodes, then connect races to circuits and seasons."""
    entity_specs = (
        ("drivers", "Driver", "driverId"),
        ("constructors", "Constructor", "constructorId"),
        ("circuits", "Circuit", "circuitId"),
        ("seasons", "Season", "year"),
        ("status", "Status", "statusId"),
        ("races", "Race", "raceId"),
    )
    for csv_name, label, key in entity_specs:
        rows = entity_rows(csv_name)
        session.execute_write(load_entity_batch, label, key, rows)

    race_relationships = [
        {
            "raceId": integer(row["raceId"]),
            "circuitId": integer(row["circuitId"]),
            "year": integer(row["year"]),
        }
        for row in csv_rows("races")
    ]
    session.execute_write(load_race_relationships, race_relationships)


def result_rows() -> list[dict[str, Any]]:
    """Build typed Result fact-node properties from results.csv."""
    fields = (
        "resultId",
        "raceId",
        "driverId",
        "constructorId",
        "number",
        "grid",
        "position",
        "positionText",
        "positionOrder",
        "points",
        "laps",
        "time",
        "milliseconds",
        "fastestLap",
        "rank",
        "fastestLapTime",
        "fastestLapSpeed",
        "statusId",
    )
    integer_fields = {
        "resultId",
        "raceId",
        "driverId",
        "constructorId",
        "number",
        "grid",
        "position",
        "positionOrder",
        "laps",
        "milliseconds",
        "fastestLap",
        "rank",
        "statusId",
    }
    float_fields = {"points", "fastestLapSpeed"}
    rows: list[dict[str, Any]] = []
    for source in csv_rows("results"):
        properties: dict[str, Any] = {}
        for field in fields:
            value = source[field]
            if field in integer_fields:
                properties[field] = integer(value)
            elif field in float_fields:
                properties[field] = decimal(value)
            else:
                properties[field] = value
        rows.append(omit_missing(properties))
    return rows


def load_result_batch(tx: Any, rows: list[dict[str, Any]]) -> None:
    """Merge Result nodes and their required relationships."""
    tx.run(
        "UNWIND $rows AS row "
        "MERGE (result:Result {resultId: row.resultId}) "
        "SET result += row "
        "WITH result, row "
        "MATCH (race:Race {raceId: row.raceId}) "
        "MATCH (driver:Driver {driverId: row.driverId}) "
        "MATCH (constructor:Constructor {constructorId: row.constructorId}) "
        "MATCH (status:Status {statusId: row.statusId}) "
        "MERGE (result)-[:FOR_RACE]->(race) "
        "MERGE (result)-[:BY_DRIVER]->(driver) "
        "MERGE (result)-[:FOR_CONSTRUCTOR]->(constructor) "
        "MERGE (result)-[:HAS_STATUS]->(status)",
        rows=rows,
    )


def load_results(session: Any) -> None:
    """Load Result fact nodes after all parent nodes exist."""
    session.execute_write(load_result_batch, result_rows())


def qualifying_rows() -> list[dict[str, Any]]:
    """Build typed Qualifying fact-node properties from qualifying.csv."""
    integer_fields = {
        "qualifyId",
        "raceId",
        "driverId",
        "constructorId",
        "number",
        "position",
    }
    fields = (
        "qualifyId",
        "raceId",
        "driverId",
        "constructorId",
        "number",
        "position",
        "q1",
        "q2",
        "q3",
    )
    rows: list[dict[str, Any]] = []
    for source in csv_rows("qualifying"):
        properties = {
            field: integer(source[field]) if field in integer_fields else source[field]
            for field in fields
        }
        rows.append(omit_missing(properties))
    return rows


def load_qualifying_batch(tx: Any, rows: list[dict[str, Any]]) -> None:
    """Merge Qualifying nodes and their required relationships."""
    tx.run(
        "UNWIND $rows AS row "
        "MERGE (qualifying:Qualifying {qualifyId: row.qualifyId}) "
        "SET qualifying += row "
        "WITH qualifying, row "
        "MATCH (race:Race {raceId: row.raceId}) "
        "MATCH (driver:Driver {driverId: row.driverId}) "
        "MATCH (constructor:Constructor {constructorId: row.constructorId}) "
        "MERGE (qualifying)-[:FOR_RACE]->(race) "
        "MERGE (qualifying)-[:BY_DRIVER]->(driver) "
        "MERGE (qualifying)-[:FOR_CONSTRUCTOR]->(constructor)",
        rows=rows,
    )


def load_qualifying(session: Any) -> None:
    """Load Qualifying fact nodes after all parent nodes exist."""
    session.execute_write(load_qualifying_batch, qualifying_rows())


def pit_stop_rows() -> list[dict[str, Any]]:
    """Build typed PitStop fact-node properties and composite event identifiers."""
    integer_fields = {"raceId", "driverId", "stop", "lap", "milliseconds"}
    fields = ("raceId", "driverId", "stop", "lap", "time", "duration", "milliseconds")
    rows: list[dict[str, Any]] = []
    for source in csv_rows("pit_stops"):
        race_id = integer(source["raceId"])
        driver_id = integer(source["driverId"])
        stop = integer(source["stop"])
        properties: dict[str, Any] = {
            "eventId": f"{race_id}:{driver_id}:{stop}",
        }
        for field in fields:
            properties[field] = (
                integer(source[field]) if field in integer_fields else source[field]
            )
        rows.append(omit_missing(properties))
    return rows


def load_pit_stop_batch(tx: Any, rows: list[dict[str, Any]]) -> None:
    """Merge PitStop nodes and their required relationships."""
    tx.run(
        "UNWIND $rows AS row "
        "MERGE (pitStop:PitStop {eventId: row.eventId}) "
        "SET pitStop += row "
        "WITH pitStop, row "
        "MATCH (race:Race {raceId: row.raceId}) "
        "MATCH (driver:Driver {driverId: row.driverId}) "
        "MERGE (pitStop)-[:FOR_RACE]->(race) "
        "MERGE (pitStop)-[:BY_DRIVER]->(driver)",
        rows=rows,
    )


def load_pit_stops(session: Any) -> None:
    """Load PitStop fact nodes after all parent nodes exist."""
    session.execute_write(load_pit_stop_batch, pit_stop_rows())


def graph_counts(tx: Any) -> dict[str, int]:
    """Return counts used to verify the parent-node loading phase."""
    result = tx.run(
        "CALL () { MATCH (node) RETURN count(node) AS nodes } "
        "CALL () { MATCH ()-[relationship]->() RETURN count(relationship) AS relationships } "
        "RETURN nodes, relationships"
    ).single()
    return {
        "nodes": result["nodes"],
        "relationships": result["relationships"],
    }


def main() -> None:
    settings = read_config()
    driver = neo4j_driver(settings)
    database = settings.get("DATABASE")
    try:
        with driver.session(database=database) as session:
            session.execute_write(create_constraints)
            load_entities(session)
            load_results(session)
            load_qualifying(session)
            load_pit_stops(session)
            counts = session.execute_read(graph_counts)
        print("Neo4j connection verified and schema constraints are ready.")
        print(f"CSV source: {CSV_DIR}")
        print(
            "Parent graph loaded: "
            f"{counts['nodes']} nodes, {counts['relationships']} relationships."
        )
    finally:
        driver.close()


if __name__ == "__main__":
    main()
