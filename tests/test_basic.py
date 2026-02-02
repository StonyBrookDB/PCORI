"""Basic tests for PCORI project structure and imports."""
import os
import pytest


class TestProjectStructure:
    """Test that project structure is correct."""

    def test_readme_exists(self):
        """README.md should exist in project root."""
        assert os.path.exists('README.md')

    def test_license_exists(self):
        """LICENSE should exist in project root."""
        assert os.path.exists('LICENSE')

    def test_requirements_exists(self):
        """requirements.txt should exist."""
        assert os.path.exists('requirements.txt')

    def test_docs_directory(self):
        """docs/ directory should contain key documentation."""
        assert os.path.isdir('docs')
        assert os.path.exists('docs/ARCHITECTURE.md')
        assert os.path.exists('docs/INSTALLATION.md')

    def test_sitl_dashboard_exists(self):
        """SITL Dashboard component should exist."""
        assert os.path.isdir('SITL_Dashboard_PCORI')
        assert os.path.exists('SITL_Dashboard_PCORI/README.md')

    def test_pipeline_exists(self):
        """Pipeline component should exist."""
        assert os.path.isdir('pipeline-pcori')
        assert os.path.exists('pipeline-pcori/README.md')


class TestReadmeContent:
    """Test that README contains required sections."""

    @pytest.fixture
    def readme_content(self):
        with open('README.md', 'r') as f:
            return f.read()

    def test_has_project_overview(self, readme_content):
        assert '## Project Overview' in readme_content

    def test_has_compliance_section(self, readme_content):
        assert '## Research Compliance' in readme_content

    def test_has_license_section(self, readme_content):
        assert '## License' in readme_content

    def test_has_citation(self, readme_content):
        assert '## Citation' in readme_content
