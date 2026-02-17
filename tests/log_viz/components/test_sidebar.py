"""Tests for log_viz/components/sidebar.py."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock streamlit before importing sidebar
mock_streamlit = MagicMock()
mock_streamlit.cache_data = lambda **kwargs: lambda func: func  # noqa: ARG005
mock_streamlit.cache_resource = lambda: lambda func: func
sys.modules["streamlit"] = mock_streamlit

from log_viz.components.sidebar import render_sidebar


class TestRenderSidebar:
    """Tests for render_sidebar function."""

    def test_returns_correct_dict_shape(self):
        """Test that render_sidebar returns dict with expected keys."""
        mock_loader = MagicMock()
        mock_loader.get_available_runs.return_value = ["trials_20260217_100000"]
        mock_loader.get_run_metadata.return_value = {
            "timestamp": "2026-02-17",
            "status": "completed",
        }

        # Mock st components used inside render_sidebar
        with patch("log_viz.components.sidebar.st") as mock_st:
            mock_st.sidebar.__enter__ = MagicMock(return_value=None)
            mock_st.sidebar.__exit__ = MagicMock(return_value=False)
            mock_st.checkbox.side_effect = [
                True,
                False,
            ]  # auto_refresh, show_historical
            mock_st.slider.return_value = 2
            mock_st.selectbox.return_value = "trials_20260217_100000"
            # Use MagicMock for session_state to support attribute-style access
            mock_session = MagicMock()
            mock_session.__contains__ = MagicMock(return_value=False)
            mock_st.session_state = mock_session

            result = render_sidebar(mock_loader)

        assert "selected_run" in result
        assert "auto_refresh" in result
        assert "refresh_interval" in result
        assert "show_historical" in result
        assert result["selected_run"] == "trials_20260217_100000"

    def test_returns_none_when_no_runs(self):
        """Test that selected_run is None when no runs are available."""
        mock_loader = MagicMock()
        mock_loader.get_available_runs.return_value = []

        with patch("log_viz.components.sidebar.st") as mock_st:
            mock_st.sidebar.__enter__ = MagicMock(return_value=None)
            mock_st.sidebar.__exit__ = MagicMock(return_value=False)
            mock_st.checkbox.return_value = True
            mock_st.slider.return_value = 2
            mock_session = MagicMock()
            mock_session.__contains__ = MagicMock(return_value=False)
            mock_st.session_state = mock_session

            result = render_sidebar(mock_loader)

        assert result["selected_run"] is None
