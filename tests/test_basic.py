"""Basic tests for PCORI infrastructure."""

import pytest
import os


class TestProjectStructure:
    """Test that required project files exist."""

    def test_readme_exists(self):
        assert os.path.exists('README.md')

    def test_license_exists(self):
        assert os.path.exists('LICENSE')

    def test_contributing_exists(self):
        assert os.path.exists('CONTRIBUTING.md')

    def test_requirements_exists(self):
        assert os.path.exists('requirements.txt')


class TestDocumentation:
    """Test that documentation files exist."""

    def test_architecture_doc(self):
        assert os.path.exists('docs/ARCHITECTURE.md')

    def test_installation_doc(self):
        assert os.path.exists('docs/INSTALLATION.md')

    def test_api_doc(self):
        assert os.path.exists('docs/API.md')


class TestComponents:
    """Test that main components exist."""

    def test_sitl_dashboard_exists(self):
        assert os.path.isdir('SITL_Dashboard_PCORI')

    def test_pipeline_exists(self):
        assert os.path.isdir('pipeline-pcori')

    def test_feature_selection_exists(self):
        assert os.path.isdir('feature_selection')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
