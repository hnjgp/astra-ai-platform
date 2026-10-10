from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return (ROOT / name).read_text(encoding="utf-8")


def test_dockerfile_runs_as_non_root_with_healthcheck():
    dockerfile = read("Dockerfile")

    assert "USER astra" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "/health" in dockerfile


def test_secrets_and_local_files_stay_out_of_the_image():
    ignored = read(".dockerignore").splitlines()

    for entry in [".env", "aivenv/", "*.db", "tests/", ".git/"]:
        assert entry in ignored


def test_compose_runs_api_with_healthy_database():
    compose = yaml.safe_load(read("docker-compose.yml"))

    api = compose["services"]["api"]
    postgres = compose["services"]["postgres"]

    assert api["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert "pg_isready" in " ".join(postgres["healthcheck"]["test"])
    assert "@postgres:5432/" in api["environment"]["RAG_DATABASE_URL"]
    assert "@postgres:5432/" in api["environment"]["DATABASE_URL"]


def test_requirements_have_no_duplicates_or_test_tools():
    packages = [
        line.split("==")[0].lower()
        for line in read("requirements.txt").splitlines()
        if line.strip()
    ]

    assert len(packages) == len(set(packages))
    assert "pytest" not in packages
    assert "-r requirements.txt" in read("requirements-dev.txt")
