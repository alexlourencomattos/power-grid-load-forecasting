import unittest

import pandas as pd

from ons_load_pipeline.transform import build_gold, normalize_csv


def csv_bytes(rows: list[tuple[str, str, str, str]]) -> bytes:
    lines = ["id_subsistema;nom_subsistema;din_instante;val_cargaenergiahomwmed"]
    lines.extend(";".join(row) for row in rows)
    return ("\n".join(lines) + "\n").encode()


def valid_rows() -> list[tuple[str, str, str, str]]:
    return [
        (code, name, timestamp, value)
        for timestamp in ("2025-01-01 00:00:00", "2025-01-01 01:00:00")
        for code, name, value in (
            ("N", "Norte", "1000,0"),
            ("NE", "Nordeste", "2000,0"),
            ("S", "Sul", "3000,0"),
            ("SE", "Sudeste/Centro-Oeste", "4000,0"),
        )
    ]


class TransformTests(unittest.TestCase):
    def test_normalize_and_build_sin_total(self) -> None:
        silver = normalize_csv(csv_bytes(valid_rows()), 2025)
        gold = build_gold(silver)
        self.assertEqual(len(silver), 8)
        self.assertEqual(list(gold["load_sin_mwmed"]), [10000.0, 10000.0])
        self.assertEqual(
            set(gold.columns),
            {
                "timestamp", "load_n_mwmed", "load_ne_mwmed", "load_s_mwmed",
                "load_se_mwmed", "load_sin_mwmed",
            },
        )

    def test_rejects_duplicate_key(self) -> None:
        rows = valid_rows()
        with self.assertRaisesRegex(ValueError, "duplicadas"):
            normalize_csv(csv_bytes(rows + [rows[0]]), 2025)

    def test_gold_requires_all_subsystems_for_sin(self) -> None:
        silver = normalize_csv(csv_bytes(valid_rows()), 2025)
        silver = silver.loc[
            ~((silver["subsystem"] == "N") & (silver["timestamp"].dt.hour == 0))
        ]
        gold = build_gold(silver)
        self.assertTrue(pd.isna(gold.loc[0, "load_sin_mwmed"]))


if __name__ == "__main__":
    unittest.main()

