import csv
import tempfile
import unittest
from pathlib import Path

from sdss_point_benchmark.cli import main
from sdss_point_benchmark.io import load_source_catalog
from sdss_point_benchmark.sdss_dr17 import (
    build_sdss_field_manifest,
    load_sdss_source_catalog,
    write_field_manifest,
)


class SdssDr17AdapterTests(unittest.TestCase):
    def test_build_manifest_groups_complete_ugriz_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest_frames.csv").write_text(
                "run,rerun,camcol,field,band,status,bytes,path,url,error\n"
                "1302,301,2,100,u,exists,1,/data/frame-u.fits.bz2,,\n"
                "1302,301,2,100,g,exists,1,/data/frame-g.fits.bz2,,\n"
                "1302,301,2,100,r,exists,1,/data/frame-r.fits.bz2,,\n"
                "1302,301,2,100,i,exists,1,/data/frame-i.fits.bz2,,\n"
                "1302,301,2,100,z,exists,1,/data/frame-z.fits.bz2,,\n",
                encoding="utf-8",
            )
            (root / "manifest_catalogs.csv").write_text(
                "run,rerun,camcol,field,status,n_objects,path,error\n"
                "1302,301,2,100,downloaded,42,/data/catalog.csv,\n",
                encoding="utf-8",
            )

            records = build_sdss_field_manifest(root)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].field_id, "run001302_camcol2_field0100")
        self.assertEqual(records[0].status, "ready")
        self.assertEqual(records[0].frame_paths["r"], "/data/frame-r.fits.bz2")
        self.assertEqual(records[0].catalog_path, "/data/catalog.csv")
        self.assertEqual(records[0].n_objects, 42)

    def test_write_manifest_csv_uses_stable_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest_frames.csv").write_text(
                "run,rerun,camcol,field,band,status,bytes,path,url,error\n"
                "1302,301,2,100,u,exists,1,/data/u,,\n"
                "1302,301,2,100,g,exists,1,/data/g,,\n"
                "1302,301,2,100,r,exists,1,/data/r,,\n"
                "1302,301,2,100,i,exists,1,/data/i,,\n"
                "1302,301,2,100,z,exists,1,/data/z,,\n",
                encoding="utf-8",
            )
            (root / "manifest_catalogs.csv").write_text(
                "run,rerun,camcol,field,status,n_objects,path,error\n"
                "1302,301,2,100,downloaded,42,/data/catalog.csv,\n",
                encoding="utf-8",
            )
            output = root / "field_manifest.csv"

            write_field_manifest(build_sdss_field_manifest(root), output)

            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["field_id"], "run001302_camcol2_field0100")
        self.assertEqual(rows[0]["frame_u"], "/data/u")
        self.assertEqual(rows[0]["frame_z"], "/data/z")

    def test_load_sdss_catalog_maps_photoobj_fields_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog_run001302_camcol2_field0100.csv"
            path.write_text(
                "objID,run,rerun,camcol,field,ra,dec,l,b,type_name,clean,rowc_r,colc_r,"
                "psfMag_u,psfMag_g,psfMag_r,psfMag_i,psfMag_z,"
                "cModelMag_u,cModelMag_g,cModelMag_r,cModelMag_i,cModelMag_z,"
                "petroR50_r,expAB_r,flags\n"
                "123,1302,301,2,100,121.1,42.5,177.5,30.8,STAR,1,20.5,10.5,"
                "17.0,16.0,15.0,14.8,14.7,"
                "-9999,-9999,-9999,-9999,-9999,"
                "1.2,0.8,64\n"
                "124,1302,301,2,100,121.2,42.6,177.6,30.9,GALAXY,1,30.5,40.5,"
                "-9999,-9999,-9999,-9999,-9999,"
                "20.0,19.0,18.0,17.8,17.7,"
                "2.4,0.5,128\n"
                "125,1302,301,2,100,121.3,42.7,177.7,31.0,OTHER,1,35.0,45.0,"
                "-9999,-9999,-9999,-9999,-9999,"
                "-9999,-9999,-9999,-9999,-9999,"
                "-9999,-9999,256\n",
                encoding="utf-8",
            )

            records = load_sdss_source_catalog(path)

        self.assertEqual([record.source_id for record in records], ["123", "124"])
        self.assertEqual(records[0].label, "star")
        self.assertEqual(records[0].x, 10.5)
        self.assertEqual(records[0].y, 20.5)
        self.assertEqual(records[0].mag_r, 15.0)
        self.assertEqual(records[1].label, "galaxy")
        self.assertEqual(records[1].mag_r, 18.0)
        self.assertEqual(records[1].size, 2.4)
        self.assertAlmostEqual(records[1].ellipticity, 0.5)
        self.assertEqual(records[1].galactic_latitude, 30.9)
        self.assertEqual(records[1].region_id, "run001302_camcol2_field0100")

    def test_cli_convert_sdss_catalog_writes_benchmark_source_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            photoobj = Path(tmp) / "catalog_run001302_camcol2_field0100.csv"
            output = Path(tmp) / "source_catalog.csv"
            photoobj.write_text(
                "objID,run,rerun,camcol,field,ra,dec,l,b,type_name,clean,rowc_r,colc_r,"
                "psfMag_u,psfMag_g,psfMag_r,psfMag_i,psfMag_z,"
                "cModelMag_u,cModelMag_g,cModelMag_r,cModelMag_i,cModelMag_z,"
                "petroR50_r,expAB_r,flags\n"
                "123,1302,301,2,100,121.1,42.5,177.5,30.8,STAR,1,20.5,10.5,"
                "17.0,16.0,15.0,14.8,14.7,"
                "-9999,-9999,-9999,-9999,-9999,"
                "1.2,0.8,64\n",
                encoding="utf-8",
            )

            exit_code = main(["convert-sdss-catalog", "--input", str(photoobj), "--output", str(output)])
            records = load_source_catalog(output)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source_id, "123")
        self.assertEqual(records[0].cutout_id, "run001302_camcol2_field0100")
        self.assertEqual(records[0].label, "star")
        self.assertEqual(records[0].mag_r, 15.0)


if __name__ == "__main__":
    unittest.main()
