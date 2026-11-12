from click.testing import CliRunner

from ncitools.cli import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "geom" in result.output


def test_cli_geom_runs_on_water_dimer(tmp_xyz, water_dimer, tmp_path):
    runner = CliRunner()
    xyz = tmp_xyz(water_dimer, "water.xyz")

    result = runner.invoke(cli, ["geom", xyz, "-o", str(tmp_path / "out")])
    assert result.exit_code == 0
    assert "Hydrogen bonds" in result.output
    assert (tmp_path / "out.nci").exists()


def test_cli_geom_skip_hb(tmp_xyz, water_dimer, tmp_path):
    runner = CliRunner()
    xyz = tmp_xyz(water_dimer, "water.xyz")

    result = runner.invoke(
        cli,
        ["geom", xyz, "--skip", "HB", "-o", str(tmp_path / "out")],
    )
    assert result.exit_code == 0
    # The HB section must be gone …
    assert "Hydrogen bonds" not in result.output
    # … but the run must still have completed and written a report.
    assert (tmp_path / "out.nci").exists()


def test_cli_geom_without_clash_flag(
    tmp_xyz, clashing_h2_dimer, tmp_path
):
    runner = CliRunner()
    xyz = tmp_xyz(clashing_h2_dimer, "clash.xyz")

    result = runner.invoke(cli, ["geom", xyz, "-o", str(tmp_path / "out")])
    assert result.exit_code == 0
    assert "spatial proximity warning" not in result.output.lower()
    assert (tmp_path / "out.nci").exists()


def test_cli_geom_with_clash_flag_creates_warnings_file(
    tmp_xyz, clashing_h2_dimer, tmp_path
):
    runner = CliRunner()
    xyz = tmp_xyz(clashing_h2_dimer, "clash.xyz")

    result = runner.invoke(
        cli, ["geom", xyz, "--clash", "-o", str(tmp_path / "out")]
    )
    assert result.exit_code == 0
    assert (tmp_path / "out_warnings.nci").exists()