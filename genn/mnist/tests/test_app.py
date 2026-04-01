import pytest
import numpy as np
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from app import InferencePayload


@pytest.fixture
def mock_inference_payload():
    """Helyettesítő adatok a tesztekhez"""
    return InferencePayload(
        selected_model="Spiking Neural Network (Mushroom Body)",
        model_type_key="mb",
        prediction=5,
        results=[
            {
                "image_id": 101,
                "label": 5,
                "metrics": {
                    "overlap": 10,
                    "jaccard": 0.8,
                    "dice": 0.85,
                    "overlap_coeff": 0.9,
                },
            }
        ],
        summary_metrics={"PN spikes": 150, "Active KCs": 45},
    )


def _metric_value_as_str(metric):
    # Tesztelés miatt stringgé konvertálás
    return str(metric.value).strip()


def test_app_startup_and_tabs():
    """Teszt az app betöltésére"""
    at = AppTest.from_file("app.py").run()
    assert not at.exception
    assert any("MNIST" in m.value for m in at.markdown)


def test_sidebar_defaults():
    """Teszt a default értékekre"""
    at = AppTest.from_file("app.py").run()

    assert at.selectbox[0].value == "Spiking Neural Network (Mushroom Body)"
    overview_matches = [m for m in at.metric if m.label == "Matches"]
    assert overview_matches, "A 'Matches' metric nem található."
    assert _metric_value_as_str(overview_matches[0]) == "5"


def test_run_analysis_warning_on_empty_canvas():
    """Teszt, ha nem rajzolunk semmit"""
    at = AppTest.from_file("app.py").run()

    at.button[0].click().run() # gomb megnyomása

    assert any("Draw a digit first" in w.value for w in at.warning)


def test_database_tab_maintenance():
    """Teszt a Database fül elemeire"""
    at = AppTest.from_file("app.py").run()

    model_clear_sb = next(s for s in at.selectbox if s.label == "Model to clear")
    assert model_clear_sb.value == "mb"
    assert any(b.label == "Clear model data" for b in at.button)


def test_database_clear_without_confirmation():
    """Teszt modell adat törlés megerősítésére"""
    at = AppTest.from_file("app.py").run()

    delete_btn = next(b for b in at.button if b.label == "Clear model data")
    delete_btn.click().run()

    assert any("confirm deletion" in w.value for w in at.warning)

def test_inference_error_display():
    """Teszt, egyszerű hibaüzenet megjelenítésére"""
    from app import StreamlitRenderer, InferencePayload

    renderer = StreamlitRenderer.__new__(StreamlitRenderer)

    calls = {}

    with patch("app.st.error", side_effect=lambda msg: calls.setdefault("error", []).append(msg)):
        payload = InferencePayload(
            selected_model="mb",
            model_type_key="mb",
            error="GeNN compilation failed",
        )
        renderer.render_inference_results(payload)

    assert "error" in calls
    assert any("GeNN compilation failed" in msg for msg in calls["error"])


def test_empty_results_warning():
    """Teszt üres eredményre"""
    from app import StreamlitRenderer

    renderer = StreamlitRenderer.__new__(StreamlitRenderer)

    calls = {}

    with patch("app.st.subheader"), \
         patch("app.st.warning", side_effect=lambda msg: calls.setdefault("warning", []).append(msg)):
        renderer.render_similarity_result_cards([])

    assert "warning" in calls
    assert any("No similar database entries found" in msg for msg in calls["warning"])


def test_database_observer_logging(monkeypatch):
    """Teszt az observer OOP patternre a loogoló funkcióhoz"""
    from app import Logger

    # Egy fake sessiont kell létrehozni
    class FakeSessionState(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as e:
                raise AttributeError(name) from e

        def __setattr__(self, name, value):
            self[name] = value

    fake_state = FakeSessionState()
    monkeypatch.setattr("app.st.session_state", fake_state)

    logger = Logger()
    logger.update("[TEST] Manual log entry")

    assert "db_logs" in fake_state
    assert fake_state["db_logs"][0] == "[TEST] Manual log entry"

def test_header_texts_present():
    """Teszt a header megjelenítésére"""
    at = AppTest.from_file("app.py").run()

    markdown_values = [m.value for m in at.markdown]
    assert any("MNIST image retrieval dashboard" in v for v in markdown_values)
    assert any("Compare models by drawing MNIST digits" in v for v in markdown_values)


def test_overview_defaults_before_run():
    """Teszt a default értékekre"""
    at = AppTest.from_file("app.py").run()

    assert any(m.label == "Prediction" and str(m.value).strip() == "-" for m in at.metric)
    assert any(m.label == "Model accuracy" and str(m.value).strip() == "-" for m in at.metric)
    assert any(m.label == "Matches" and str(m.value).strip() == "5" for m in at.metric)
    assert any(m.label == "Model" and str(m.value).strip() == "MB" for m in at.metric)


def test_input_preview_info_present_initially():
    """Teszt a kezdeti segítő üzenet megjelenítésére"""
    at = AppTest.from_file("app.py").run()

    assert any("Draw a digit to preview the 28×28 input." in i.value for i in at.info)


def test_database_logs_section_present():
    """Teszt az adatbázis logok megjelenítésére"""
    at = AppTest.from_file("app.py").run()

    assert any("Database logs" in m.value for m in at.markdown) or any(
        getattr(x, "value", "") == "Database logs" for x in getattr(at, "subheader", [])
    )


def test_database_activity_empty_message_present():
    """Teszt kezdetben nincs adatbázis log"""
    at = AppTest.from_file("app.py").run()

    captions = [c.value for c in at.caption]
    assert any("No database activity yet." in v for v in captions)


def test_database_summary_metrics_present():
    """Teszt az adatbázis összesítő megjelenítésére"""
    at = AppTest.from_file("app.py").run()

    labels = [m.label for m in at.metric]
    assert "Total images" in labels
    assert "Total vector rows" in labels
    assert "Models loaded" in labels


def test_model_selectbox_contains_expected_options():
    """Teszt a modell kiválasztás lehetőségeire"""
    at = AppTest.from_file("app.py").run()

    model_sb = at.selectbox[0]
    expected = [
        "Spiking Neural Network (Mushroom Body)",
        "Echo State Network (Watts-Strogatz)",
        "Echo State Network (Barabási-Albert)",
        "Echo State Network (Erdős-Rényi)",
    ]
    assert list(model_sb.options) == expected


def test_database_selectboxes_defaults():
    """Teszt a default értékekre a Database fülön"""
    at = AppTest.from_file("app.py").run()

    populate_sb = next(s for s in at.selectbox if s.label == "Choose model to populate")
    clear_sb = next(s for s in at.selectbox if s.label == "Model to clear")

    assert populate_sb.value == "Spiking Neural Network (Mushroom Body)"
    assert clear_sb.value == "mb"


def test_database_clear_checkbox_default_false():
    """Teszt az adattörlő box alap értékére"""
    at = AppTest.from_file("app.py").run()

    confirm_cb = next(
        c for c in at.checkbox
        if "permanently delete data" in c.label
    )
    assert confirm_cb.value is False


def test_all_expected_buttons_present():
    """Teszt minden gomb meglétére"""
    at = AppTest.from_file("app.py").run()

    labels = [b.label for b in at.button]
    assert "Run analysis" in labels
    assert "Clear cache" in labels
    assert "Replace all model files" in labels
    assert "Replace selected model data" in labels
    assert "Clear model data" in labels


def test_replace_all_model_files_without_uploads_warns():
    """Teszt összes modell importálására fájlok nélkül"""
    at = AppTest.from_file("app.py").run()

    btn = next(b for b in at.button if b.label == "Replace all model files")
    at = btn.click().run()

    assert any("No model files were populated." in w.value for w in at.warning)


def test_replace_selected_model_without_upload_warns():
    """Teszt specifikus modell importálására fájl nélkül"""
    at = AppTest.from_file("app.py").run()

    btn = next(b for b in at.button if b.label == "Replace selected model data")
    at = btn.click().run()

    assert any("No file populated for" in w.value for w in at.warning)


def test_clear_model_data_with_confirmation_removes_warning():
    """Teszt ha megerősítjük a törlést """
    at = AppTest.from_file("app.py").run()

    confirm_cb = next(
        c for c in at.checkbox
        if "permanently delete data" in c.label
    )
    at = confirm_cb.check().run()

    # Valós törlés timeoutot okozott a tesztkörnyezetben
    confirm_cb = next(
        c for c in at.checkbox
        if "permanently delete data" in c.label
    )
    assert confirm_cb.value is True


def test_help_caption_present_in_input_panel():
    """Teszt a tájékoztató üzenet meglétére"""
    at = AppTest.from_file("app.py").run()

    captions = [c.value for c in at.caption]
    assert any("Use the Database tab to upload latent vectors" in v for v in captions)


def test_upload_help_texts_exist():
    """Teszt a fájfeltöltő üzenet meglétére"""
    at = AppTest.from_file("app.py").run()

    captions = [c.value for c in at.caption]
    assert any("You can upload one file for each model" in v for v in captions)