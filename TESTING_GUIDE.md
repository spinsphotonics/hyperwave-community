# Testing Guide - hyperwave-community

This guide covers testing the Python client library that users interact with to run simulations on the Hyperwave cloud platform.

---

## Overview

**hyperwave-community** is the client library that:
- Provides high-level API for structure creation
- Encodes/decodes simulation data
- Communicates with hyperwave-cloud API gateway
- Processes and visualizes results

---

## Test Structure

### Recommended Directory Structure

```
hyperwave-community/
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Shared pytest fixtures
│   ├── test_api_client.py           # API communication
│   ├── test_structure.py            # Structure building
│   ├── test_monitors.py             # Monitor placement
│   ├── test_sources.py              # Source creation
│   ├── test_data_serialization.py   # Recipe extraction
│   ├── test_absorption.py           # Absorber creation
│   └── integration/
│       ├── test_simulate_flow.py    # Full simulation workflow
│       └── test_api_integration.py  # Real API calls
└── hyperwave_community/
    ├── __init__.py
    ├── api_client.py
    ├── structure.py
    ├── monitors.py
    ├── sources.py
    └── ...
```

---

## 1. Unit Tests (Pure Functions, No API Calls)

### 1.1 Structure Building Tests

**File:** `tests/test_structure.py`

```python
"""
Test structure creation and manipulation.

These tests use local JAX operations only - no API calls.
"""

import pytest
import jax.numpy as jnp
import hyperwave_community as hwc


class TestDensityFiltering:
    """Test density filtering operations."""

    def test_density_creates_jax_array(self):
        """Density should return JAX array."""
        theta = jnp.zeros((100, 100))
        result = hwc.density(theta=theta, radius=8, alpha=0)

        assert isinstance(result, jnp.ndarray)
        assert result.shape == theta.shape

    def test_density_preserves_binary_values(self):
        """With alpha=0, should preserve 0 and 1 values."""
        theta = jnp.array([[0, 1], [1, 0]])
        result = hwc.density(theta=theta, radius=0, alpha=0)

        # Should be exactly 0 or 1
        assert jnp.all((result == 0) | (result == 1))

    def test_density_smooths_edges(self):
        """With radius>0, should smooth edges."""
        theta = jnp.zeros((100, 100))
        theta = theta.at[40:60, 40:60].set(1.0)

        result = hwc.density(theta=theta, radius=8, alpha=0)

        # Check that edges are smoothed (not exactly 0 or 1)
        edge_point = result[39, 50]  # Just outside square
        assert 0 < edge_point < 1


class TestLayerCreation:
    """Test Layer dataclass."""

    def test_layer_requires_density_and_permittivity(self):
        """Layer must have density and permittivity."""
        density = jnp.ones((100, 100))

        layer = hwc.Layer(
            density_pattern=density,
            permittivity_values=(2.1025, 11.56),
            layer_thickness=20
        )

        assert layer.layer_thickness == 20

    def test_layer_with_uniform_permittivity(self):
        """Layer can have single permittivity value."""
        density = jnp.ones((100, 100))

        layer = hwc.Layer(
            density_pattern=density,
            permittivity_values=2.1025,  # Single value
            layer_thickness=40
        )

        assert isinstance(layer.permittivity_values, (int, float))


class TestStructureCreation:
    """Test 3D structure building."""

    def test_create_structure_from_layers(self):
        """Should create 3D structure from layer stack."""
        density = jnp.ones((100, 100))

        layer = hwc.Layer(
            density_pattern=density,
            permittivity_values=(2.1025, 11.56),
            layer_thickness=20
        )

        structure = hwc.create_structure(
            layers=[layer],
            vertical_radius=2
        )

        # Check 3D shape
        assert len(structure.permittivity.shape) == 4  # (freq, x, y, z)
        assert structure.permittivity.shape[3] > 0  # Has z-dimension

    def test_structure_has_recipe(self):
        """Structure should have recipe attribute."""
        density = jnp.ones((50, 50))
        layer = hwc.Layer(
            density_pattern=density,
            permittivity_values=11.56,
            layer_thickness=20
        )

        structure = hwc.create_structure(layers=[layer], vertical_radius=2)

        assert hasattr(structure, 'recipe')
        assert structure.recipe is not None


class TestRecipeExtraction:
    """Test structure recipe serialization."""

    def test_recipe_size_under_20mb(self):
        """Recipe should be compact (<20MB for typical structure)."""
        import json

        # Create moderately sized structure
        density = jnp.ones((500, 500))
        layer = hwc.Layer(
            density_pattern=density,
            permittivity_values=11.56,
            layer_thickness=20
        )

        structure = hwc.create_structure(layers=[layer], vertical_radius=2)
        recipe = structure.extract_recipe()

        recipe_json = json.dumps(recipe)
        size_mb = len(recipe_json) / (1024 * 1024)

        assert size_mb < 20, f"Recipe too large: {size_mb:.1f}MB"

    def test_uniform_layer_stored_as_scalar(self):
        """Uniform density layers should be stored as scalar."""
        density = jnp.ones((100, 100))  # All ones = uniform
        layer = hwc.Layer(
            density_pattern=density,
            permittivity_values=11.56,
            layer_thickness=20
        )

        structure = hwc.create_structure(layers=[layer], vertical_radius=2)
        recipe = structure.extract_recipe()

        # Check first layer is stored as uniform
        first_layer_density = recipe['layers'][0]['density_pattern']
        assert first_layer_density['type'] == 'uniform'
        assert 'value' in first_layer_density
        assert 'shape' in first_layer_density

    def test_recipe_reconstruction(self):
        """Should be able to reconstruct structure from recipe."""
        # Create original structure
        density = jnp.zeros((100, 100))
        density = density.at[40:60, 40:60].set(1.0)

        layer = hwc.Layer(
            density_pattern=density,
            permittivity_values=(2.1025, 11.56),
            layer_thickness=20
        )

        original = hwc.create_structure(layers=[layer], vertical_radius=2)

        # Extract recipe and reconstruct
        recipe = original.extract_recipe()
        reconstructed = hwc.reconstruct_structure_from_recipe(recipe)

        # Check shapes match
        assert original.permittivity.shape == reconstructed.permittivity.shape

        # Check values are close (may have small numerical differences)
        assert jnp.allclose(original.permittivity, reconstructed.permittivity, atol=1e-6)
```

### 1.2 Monitor Placement Tests

**File:** `tests/test_monitors.py`

```python
"""
Test monitor creation and placement.
"""

import pytest
import jax.numpy as jnp
import hyperwave_community as hwc


class TestMonitorSet:
    """Test MonitorSet container."""

    def test_monitor_set_creation(self):
        """Should create empty monitor set."""
        monitors = hwc.MonitorSet()

        assert len(monitors.list_monitors()) == 0

    def test_add_monitor(self):
        """Should add monitor with name."""
        monitors = hwc.MonitorSet()

        monitors.add_monitor(
            name="Input",
            shape=(10, 50, 50),
            offset=(100, 0, 0)
        )

        monitor_list = monitors.list_monitors()
        assert len(monitor_list) == 1
        assert "Input" in monitor_list


class TestAutomaticMonitorPlacement:
    """Test automatic waveguide detection and monitor placement."""

    def test_add_monitors_at_position(self):
        """Should automatically place monitors at waveguide positions."""
        # Create simple waveguide structure
        theta = jnp.zeros((500, 250))
        theta = theta.at[100:150, :].set(1.0)  # Horizontal waveguide

        density = hwc.density(theta=theta, radius=8, alpha=0)
        layer = hwc.Layer(
            density_pattern=density,
            permittivity_values=(2.1025, 11.56),
            layer_thickness=20
        )

        structure = hwc.create_structure(layers=[layer], vertical_radius=2)

        # Add monitors
        monitors = hwc.MonitorSet()
        monitors.add_monitors_at_position(
            structure=structure,
            axis='x',
            position=100,
            label='Input'
        )

        # Should have created monitors
        assert len(monitors.list_monitors()) > 0

    def test_monitor_recipe_serialization(self):
        """Monitors should serialize to recipe."""
        monitors = hwc.MonitorSet()
        monitors.add_monitor(
            name="Test",
            shape=(10, 50, 50),
            offset=(100, 0, 0)
        )

        recipe = monitors.recipe

        assert isinstance(recipe, dict)
        assert "Test" in recipe
```

### 1.3 Source Creation Tests

**File:** `tests/test_sources.py`

```python
"""
Test source creation (local operations only).
"""

import pytest
import jax.numpy as jnp
import hyperwave_community as hwc


class TestModeSource:
    """Test modal source creation."""

    def test_create_mode_source(self):
        """Should create mode source locally."""
        # Create simple waveguide
        theta = jnp.zeros((500, 250))
        theta = theta.at[100:150, :].set(1.0)

        density = hwc.density(theta=theta, radius=8, alpha=0)
        layer = hwc.Layer(
            density_pattern=density,
            permittivity_values=(2.1025, 11.56),
            layer_thickness=20
        )

        structure = hwc.create_structure(layers=[layer], vertical_radius=2)

        # Create mode source
        freq_band = (2*jnp.pi/32, 2*jnp.pi/30, 2)
        source_field, source_offset, mode_info = hwc.create_mode_source(
            structure=structure,
            freq_band=freq_band,
            mode_num=0,
            propagation_axis='x',
            source_position=80,
            perpendicular_bounds=(0, 250)
        )

        # Check source field shape
        assert source_field.shape[0] == 2  # 2 frequencies
        assert source_field.shape[1] == 6  # 6 field components

        # Check mode info
        assert 'beta' in mode_info
        assert 'error' in mode_info
```

### 1.4 Absorption Tests

**File:** `tests/test_absorption.py`

```python
"""
Test absorbing boundary creation.
"""

import pytest
import jax.numpy as jnp
import hyperwave_community as hwc


class TestAbsorptionMask:
    """Test absorption mask creation."""

    def test_create_absorption_mask(self):
        """Should create 3D absorption mask."""
        grid_shape = (500, 250, 100)

        mask = hwc.create_absorption_mask(
            grid_shape=grid_shape,
            absorption_widths=(70, 35, 17),
            absorption_coeff=4.89e-3
        )

        assert mask.shape == grid_shape

    def test_absorption_gradual_at_edges(self):
        """Absorption should be gradual (not step function)."""
        grid_shape = (100, 100, 100)

        mask = hwc.create_absorption_mask(
            grid_shape=grid_shape,
            absorption_widths=(20, 20, 20),
            absorption_coeff=1.0
        )

        # Check that absorption increases gradually
        edge_values = mask[0:20, 50, 50]  # X-edge

        # Should increase monotonically from edge
        assert jnp.all(edge_values[1:] >= edge_values[:-1])

    def test_absorption_zero_in_center(self):
        """Center of domain should have zero absorption."""
        grid_shape = (200, 200, 200)

        mask = hwc.create_absorption_mask(
            grid_shape=grid_shape,
            absorption_widths=(30, 30, 30),
            absorption_coeff=1.0
        )

        # Center should be zero
        center = mask[100, 100, 100]
        assert jnp.abs(center) < 1e-10
```

---

## 2. Integration Tests (With Mocked API)

### 2.1 API Client Tests

**File:** `tests/test_api_client.py`

```python
"""
Test API client communication without calling real API.
"""

import pytest
from unittest.mock import patch, MagicMock
import hyperwave_community as hwc


class TestAPIClient:
    """Test API communication logic."""

    @patch('requests.post')
    def test_simulate_sends_correct_request(self, mock_post):
        """Should send properly formatted request to API."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "monitor_data_b64": {},
            "monitor_data_shapes": {},
            "transmission": [0.99],
            "sim_time": 25.0
        }
        mock_post.return_value = mock_response

        # Create minimal structure
        density = jnp.ones((50, 50))
        layer = hwc.Layer(
            density_pattern=density,
            permittivity_values=11.56,
            layer_thickness=20
        )
        structure = hwc.create_structure(layers=[layer], vertical_radius=2)

        # Create source
        source_field = jnp.ones((2, 6, 1, 50, 50))
        source_offset = (80, 0, 0)

        # Call simulate (should be mocked)
        result = hwc.simulate(
            structure_recipe=structure.extract_recipe(),
            source_field=source_field,
            source_offset=source_offset,
            freq_band=(0.196, 0.209, 2),
            monitors_recipe={},
            api_key="test-key",
            gpu_type="H100"
        )

        # Verify request was made
        assert mock_post.called

        # Verify request format
        call_args = mock_post.call_args
        assert "X-API-Key" in call_args.kwargs["headers"]
        assert "structure_recipe" in call_args.kwargs["json"]

    @patch('requests.post')
    def test_simulate_handles_401_error(self, mock_post):
        """Should raise clear error on 401 Unauthorized."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"detail": "Invalid API key"}
        mock_post.return_value = mock_response

        # Should raise exception with clear message
        with pytest.raises(Exception) as exc_info:
            hwc.simulate(
                structure_recipe={},
                source_field=jnp.ones((2, 6, 1, 10, 10)),
                source_offset=(0, 0, 0),
                freq_band=(0.1, 0.2, 2),
                monitors_recipe={},
                api_key="invalid-key"
            )

        assert "401" in str(exc_info.value) or "Unauthorized" in str(exc_info.value)

    @patch('requests.post')
    def test_simulate_handles_402_insufficient_credits(self, mock_post):
        """Should raise clear error on 402 Payment Required."""
        mock_response = MagicMock()
        mock_response.status_code = 402
        mock_response.json.return_value = {"detail": "Insufficient credits"}
        mock_post.return_value = mock_response

        with pytest.raises(Exception) as exc_info:
            hwc.simulate(
                structure_recipe={},
                source_field=jnp.ones((2, 6, 1, 10, 10)),
                source_offset=(0, 0, 0),
                freq_band=(0.1, 0.2, 2),
                monitors_recipe={},
                api_key="valid-key"
            )

        assert "402" in str(exc_info.value) or "credit" in str(exc_info.value).lower()


class TestDataSerialization:
    """Test data encoding/decoding for API transport."""

    def test_source_field_base64_encoding(self):
        """Source field should encode to base64."""
        import base64
        import pickle

        source_field = jnp.ones((2, 6, 1, 50, 50))

        # Encode (simulating what API client does)
        encoded = base64.b64encode(pickle.dumps(source_field)).decode()

        # Should be string
        assert isinstance(encoded, str)

        # Should decode back to original
        decoded = pickle.loads(base64.b64decode(encoded))
        assert jnp.allclose(source_field, decoded)

    def test_monitor_data_base64_decoding(self):
        """Should decode monitor data from API response."""
        import base64
        import pickle

        # Simulate API response
        monitor_data = jnp.ones((2, 6, 10, 50, 50))
        encoded = base64.b64encode(pickle.dumps(monitor_data)).decode()

        # Decode (simulating what API client does)
        decoded = pickle.loads(base64.b64decode(encoded))

        assert jnp.allclose(monitor_data, decoded)
```

---

## 3. End-to-End Integration Tests

### 3.1 Full Simulation Flow

**File:** `tests/integration/test_simulate_flow.py`

```python
"""
Test complete simulation workflow with real API.

Requires:
- Valid API key
- Credits in account
- hyperwave-cloud API running
"""

import pytest
import os
import jax.numpy as jnp
import hyperwave_community as hwc


# Skip if no API key provided
@pytest.mark.skipif(
    os.environ.get('HYPERWAVE_API_KEY') is None,
    reason="No API key provided (set HYPERWAVE_API_KEY)"
)
class TestFullSimulationFlow:
    """Test complete simulation from structure to results."""

    def test_simple_waveguide_simulation(self):
        """Run complete simulation of simple waveguide."""
        # 1. Create structure
        theta = jnp.zeros((500, 250))
        theta = theta.at[100:150, :].set(1.0)

        density = hwc.density(theta=theta, radius=8, alpha=0)
        layer = hwc.Layer(
            density_pattern=density,
            permittivity_values=(2.1025, 11.56),
            layer_thickness=20
        )

        structure = hwc.create_structure(layers=[layer], vertical_radius=2)

        # 2. Add absorption
        Lx, Ly, Lz = structure.permittivity.shape[1:]
        absorption = hwc.create_absorption_mask(
            grid_shape=(Lx, Ly, Lz),
            absorption_widths=(70, 35, 17),
            absorption_coeff=4.89e-3
        )
        structure.conductivity = structure.conductivity + absorption

        # 3. Create source
        freq_band = (2*jnp.pi/32, 2*jnp.pi/30, 2)
        source_field, source_offset, mode_info = hwc.create_mode_source(
            structure=structure,
            freq_band=freq_band,
            mode_num=0,
            propagation_axis='x',
            source_position=80,
            perpendicular_bounds=(0, Ly)
        )

        # 4. Add monitors
        monitors = hwc.MonitorSet()
        monitors.add_monitors_at_position(
            structure=structure,
            axis='x',
            position=100,
            label='Input'
        )
        monitors.add_monitors_at_position(
            structure=structure,
            axis='x',
            position=400,
            label='Output'
        )

        # 5. Run simulation
        api_key = os.environ.get('HYPERWAVE_API_KEY')
        results = hwc.simulate(
            structure_recipe=structure.extract_recipe(),
            source_field=source_field,
            source_offset=source_offset,
            freq_band=freq_band,
            monitors_recipe=monitors.recipe,
            mode_info=mode_info,
            simulation_steps=20000,
            api_key=api_key,
            gpu_type="T4"  # Use cheapest GPU for testing
        )

        # 6. Verify results
        assert 'monitor_data' in results
        assert 'transmission' in results
        assert 'sim_time' in results

        # Check transmission is reasonable (0 to 1)
        transmission = results['transmission']
        assert jnp.all(transmission >= 0)
        assert jnp.all(transmission <= 1)

        print(f"Simulation completed in {results['sim_time']:.2f}s")
        print(f"Transmission: {jnp.mean(transmission):.4f}")
```

---

## 4. Setup Instructions

### 4.1 Install Dependencies

```bash
cd hyperwave-community

# Activate virtual environment
source .venv/bin/activate  # or create new venv

# Install package in development mode
pip install -e .

# Install test dependencies
pip install pytest pytest-cov pytest-mock

# Install optional dependencies for full testing
pip install faker factory-boy
```

### 4.2 Configure Test Environment

**File:** `tests/conftest.py`

```python
"""
Shared pytest fixtures and configuration.
"""

import pytest
import os
import jax.numpy as jnp
import hyperwave_community as hwc


@pytest.fixture
def simple_waveguide_structure():
    """Create simple waveguide structure for testing."""
    theta = jnp.zeros((500, 250))
    theta = theta.at[100:150, :].set(1.0)

    density = hwc.density(theta=theta, radius=8, alpha=0)
    layer = hwc.Layer(
        density_pattern=density,
        permittivity_values=(2.1025, 11.56),
        layer_thickness=20
    )

    return hwc.create_structure(layers=[layer], vertical_radius=2)


@pytest.fixture
def test_api_key():
    """Get API key from environment or skip test."""
    api_key = os.environ.get('HYPERWAVE_API_KEY')
    if api_key is None:
        pytest.skip("No API key provided (set HYPERWAVE_API_KEY)")
    return api_key


@pytest.fixture
def mock_api_response():
    """Standard mock API response."""
    return {
        "monitor_data_b64": {},
        "monitor_data_shapes": {},
        "monitor_names": {},
        "transmission": [0.99, 0.98],
        "powers": {},
        "sim_time": 25.0,
        "performance": 5.3e9,
        "gpu_type": "H100"
    }
```

### 4.3 Environment Variables

**File:** `.env.test`

```bash
# API Configuration
HYPERWAVE_API_KEY=your-test-api-key-here
HYPERWAVE_API_URL=https://hyperwave-api.onrender.com

# Or for local testing
# HYPERWAVE_API_URL=http://localhost:8000

# Test Mode
MOCK_API_CALLS=false  # Set to true to mock all API calls
```

---

## 5. Running Tests

### 5.1 Run All Tests

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=hyperwave_community --cov-report=html

# Run specific test file
pytest tests/test_structure.py -v
```

### 5.2 Run Unit Tests Only (Fast, No API)

```bash
# Run only unit tests (no integration)
pytest tests/test_*.py -v

# Exclude integration tests
pytest -v -m "not integration"
```

### 5.3 Run Integration Tests (Requires API Key)

```bash
# Set API key
export HYPERWAVE_API_KEY="your-key-here"

# Run integration tests
pytest tests/integration/ -v
```

---

## 6. Test Coverage Goals

| Module | Current | Target |
|--------|---------|--------|
| `structure.py` | 0% | 80%+ |
| `monitors.py` | 0% | 70%+ |
| `sources.py` | 0% | 60%+ |
| `api_client.py` | 0% | 80%+ |
| `absorption.py` | 0% | 70%+ |
| `data_io.py` | 0% | 50%+ |

---

## 7. Key Test Scenarios

### 7.1 Must Test

- [ ] Structure creation and recipe extraction
- [ ] Recipe size < 20MB for typical structures
- [ ] Mode source creation locally
- [ ] Monitor placement (manual and automatic)
- [ ] API request formatting
- [ ] API response parsing
- [ ] Error handling (401, 402, 500)

### 7.2 Should Test

- [ ] Structure reconstruction from recipe
- [ ] Complex multi-layer structures
- [ ] GDS import/export
- [ ] Absorption mask creation
- [ ] Concurrent API calls
- [ ] Network error handling

### 7.3 Nice to Test

- [ ] Very large structures (stress test)
- [ ] Edge cases in density filtering
- [ ] Monitor visualization
- [ ] Result caching

---

## 8. Common Issues

### Issue: "JAX not using GPU"

**Solution:** For tests, CPU-only JAX is sufficient. GPU not needed locally.

### Issue: "API timeout"

**Solution:** Use shorter simulations for testing, or increase timeout.

### Issue: "Recipe too large"

**Solution:** Check that uniform layers are being compressed. Debug `extract_recipe()`.

---

## 9. Next Steps

1. **Create test directory** (`mkdir tests`)
2. **Implement structure tests** (highest value)
3. **Add API client tests** (mock API calls)
4. **Add monitor tests**
5. **Measure coverage**
6. **Add integration tests** (optional, requires API key)

---

## 10. Resources

- **pytest Documentation:** https://docs.pytest.org/
- **JAX Testing:** https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
- **Mocking Guide:** https://docs.python.org/3/library/unittest.mock.html
