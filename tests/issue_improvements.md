# Proposed Improvements for FaceEnhancer Tests

## Current Test Analysis

The `tests/test_chararep_face_enhancer.py` file contains 416 lines of comprehensive tests for the FaceEnhancer class. Key observations:

### Test Coverage Strengths:
1. **Multi-backend support**: Tests both gfpgan and codeformer_onnx backends
2. **Error handling**: Extensive tests for fallback mechanisms and edge cases
3. **Configuration validation**: Tests disabled/enabled states and parameter variations

### Potential Issues and Improvements:

#### 1. Test Structure and Organization
- **PROBLEM**: Tests are organized in large classes that could be more focused
- **RECOMMENDATION**: Break `TestFaceEnhancerEnabled` into more granular test classes

#### 2. Parameter Validation
- **ISSUE**: No tests validate enhancement_model values against expected enum values
- **RECOMMENDATION**: Add tests for invalid enhancement_model values

#### 3. Performance Testing
- **ISSUE**: Missing performance tests for large frame batches
- **RECOMMENDATION**: Add benchmark tests to ensure efficient execution

#### 4. Memory Usage
- **ISSUE**: No tests for memory efficiency with large images
- **RECOMMENDATION**: Add memory usage tests for crop enhancement operations

#### 5. Integration Testing
- **ISSUE**: Tests primarily use mocked scenarios
- **RECOMMENDATION**: Add integration tests with actual model files

#### 6. Future-Proofing
- **ISSUE**: Limited tests for new enhancement models
- **RECOMMENDATION**: Create extensible test infrastructure for new backends

## Proposed Test Improvements

### 1. Optimized Test Organization
```python
class TestFaceEnhancerGfpganBackend():  # New focused class
class TestFaceEnhancerCodeformerBackend():  # New focused class  
class TestFaceEnhancerErrorScenarios():  # Consolidate error tests
```

### 2. Configuration Validation Tests
```python
def test_enhancement_model_valid_values():  # Test all supported values
def test_enhancement_model_invalid_values():  # Test error handling
```

### 3. Performance Tests
```python
def test_large_batch_enhancement():  # Test processing efficiency
def test_memory_usage_acceptable():  # Test memory consumption
```

### 4. Integration Tests
```python
@pytest.mark.integration
def test_with_actual_model_files():  # Test with real model files
```
## Best Practices Implementation

1. **Modular Test Design**: Break down monolithic test classes into focused, maintainable units
2. **Comprehensive Validation**: Ensure robust testing of configuration parameters
3. **Performance Optimization**: Include benchmark tests for large-scale scenarios
4. **Memory Management**: Add tests to verify efficient memory usage
5. **Extensible Framework**: Design tests to seamlessly support future model implementations

Recommended infrastructure improvements focus on creating a flexible, scalable testing framework that can easily adapt to new enhancement models and maintain consistent performance, memory, and integration testing strategies.<|END_THINKING|><|START_ACTION|>[
    <write_to_file>
<path>tests/test_fenhancer_edge_cases.py</path>
<content># Additional edge case tests for FaceEnhancer
\"\"\"Edge case and boundary condition tests for face_enhancer module.\"\"\"

import pytest
import numpy as np
from chararep.config import PipelineConfig\nfrom chararep.face_enhancer import FaceEnhancer\nimport types\n\n\ndef _make_cfg(enable=True, weight=0.7, model=\"gfpgan\", model_path=None) -> PipelineConfig:\n    \"\"\"Create test PipelineConfig with specified parameters.\"\"\"\n    return PipelineConfig(\n        enable_face_enhancement=enable,\n        enhancement_weight=weight,\n        enhancement_model=model,\n        enhance_model_path=model_path,\n    )\n\n\ndef _frame(h=100, w=100) -> np.ndarray:\n    \"\"\"Create a test frame.\"\"\"\n    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)\n\n\ndef _make_landmarks(x1=10, y1=10, x2=50, y2=50) -> np.ndarray:\n    \"\"\"Create test landmarks for a face.\"\"\"\n    cx = (x1 + x2) / 2.0\n    eye_y = y1 + (y2 - y1) * 0.35\n    mouth_y = y1 + (y2 - y1) * 0.72\n    return np.array(\n        [\n            [x1 + (x2 - x1) * 0.28, eye_y],\n            [x1 + (x2 - x1) * 0.72, eye_y],\n            [cx, y1 + (y2 - y1) * 0.52],\n            [x1 + (x2 - x1) * 0.34, mouth_y],\n            [x1 + (x2 - x1) * 0.66, mouth_y],\n        ],\n        dtype=np.float32,\n    )\n\n\ndef _make_tracked_face(\n    x1=10,\n    y1=10,\n    x2=50,\n    y2=50,\n    label=\"villain\",\n    landmarks=None,\n    track_id=0,\n):\n    \"\"\"Create a minimal TrackedFace-like object for testing.\"\"\"\n    return types.SimpleNamespace(\n        track_id=track_id,\n        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),\n        identity_label=label,\n        landmarks=landmarks,\n    )\n\n\nclass TestFaceEnhancerEdgeCases:\n    \"\"\"Test edge cases and boundary conditions for FaceEnhancer.\"\"\"\n\n    def test_extreme_weights(self):\n        \"\"\"Test enhancement with extreme weight values.\"\"\"\n        cfg = _make_cfg(enable=True)\n        enhancer = FaceEnhancer(cfg)\n        \n        # Test very small weight\n        cfg = _make_cfg(enable=True, weight=0.001)\n        enhancer = FaceEnhancer(cfg)\n        frame = _frame(200, 200)\n        result = enhancer.enhance(frame, frame_idx=0)\n        assert result.shape == frame.shape\n        \n        # Test very large weight (could cause overflow)\n        cfg = _make_cfg(enable=True, weight=10.0)\n        enhancer = FaceEnhancer(cfg)\n        result = enhancer.enhance(frame, frame_idx=0)\n        assert result.shape == frame.shape\n\n    def test_zero_weight_with_available_backend(self):\n        \"\"\"Test enhancement with zero weight (should be effectively disabled).\"\"\"\n        cfg = _make_cfg(enable=True, weight=0.0)\n        enhancer = FaceEnhancer(cfg)\n        assert enhancer.available\n        \n        frame = _frame(200, 200)\n        result = enhancer.enhance(frame, frame_idx=0)\n        assert result.shape == frame.shape\n\n    def test_extreme_image_dimensions(self):\n        \"\"\"Test enhancement with extreme image dimensions.\"\"\"\n        # Very small image\n        cfg = _make_cfg(enable=True)\n        enhancer = FaceEnhancer(cfg)\n        \n        small_frame = _frame(10, 10)\n        result = enhancer.enhance(small_frame, frame_idx=0)\n        assert result.shape == small_frame.shape\n        \n        # Very large image\n        large_frame = _frame(2048, 2048)\n        result = enhancer.enhance(large_frame, frame_idx=0)\n        assert result.shape == large_frame.shape\n\n    def test_extreme_bbox_coordinates(self):\n        \"\"\"Test enhancement with extreme bounding box coordinates.\"\"\"\n        cfg = _make_cfg(enable=True)\n        enhancer = FaceEnhancer(cfg)\n        \n        frame = _frame(100, 100)\n        \n        # Face at image boundary\n        face = _make_tracked_face(0, 0, 99, 99, label=\"hero\", landmarks=_make_landmarks(0, 0, 99, 99))\n        result = enhancer.enhance_faces(frame, [face], frame_idx=0)\n        assert result.shape == frame.shape\n        \n        # Face with single pixel dimensions\n        face = _make_tracked_face(50, 50, 51, 51, label=\"hero\")\n        result = enhancer.enhance_faces(frame, [face], frame_idx=0)\n        assert result.shape == frame.shape\n\n    def test_extreme_landmarks_coordinates(self):\n        \"\"\"Test enhancement with extreme landmark coordinates.\"\"\"\n        cfg = _make_cfg(enable=True)\n        enhancer = FaceEnhancer(cfg)\n        \n        frame = _frame(100, 100)\n        \n        # Landmarks at image edges\n        landmarks = np.array([\n            [0, 0],\n            [99, 0],\n            [50, 99],\n            [0, 99],\n            [99, 99],\n        ], dtype=np.float32)\n        \n        face = _make_tracked_face(0, 0, 99, 99, label=\"hero\", landmarks=landmarks)\n        result = enhancer.enhance_faces(frame, [face], frame_idx=0)\n        assert result.shape == frame.shape\n\n    def test_large_number_of_faces(self):\n        \"\"\"Test enhancement with many faces (performance/spacing issues).\"\"\"\n        cfg = _make_cfg(enable=True)\n        enhancer = FaceEnhancer(cfg)\n        \n        frame = _frame(500, 500)\n        original = frame.copy()\n        \n        # Create many faces that overlap significantly\n        faces = []\n        for i in range(10):\n            x = i * 10  # Overlapping faces\n            y = i * 10\n            face = _make_tracked_face(x, y, x + 40, y + 40, label=f\"hero_{i}\")\n            faces.append(face)\n        \n        result = enhancer.enhance_faces(frame, faces, frame_idx=0)\n        assert result.shape == frame.shape\n\n    def test_negative_weight_handling(self):\n        \"\"\"Test handling of negative weights (should be treated as positive).\"\"\"\n        cfg = _make_cfg(enable=True, weight=-0.5)\n        enhancer = FaceEnhancer(cfg)\n        assert enhancer.available\n        \n        frame = _frame(100, 100)\n        result = enhancer.enhance(frame, frame_idx=0)\n        assert result.shape == frame.shape\n\n    def test_coordinate_overflow_scenarios(self):\n        \"\"\"Test scenarios where coordinates might overflow.\"\"\"\n        cfg = _make_cfg(enable=True)\n        enhancer = FaceEnhancer(cfg)\n        \n        frame = _frame(100, 100)\n        \n        # Landmarks beyond frame dimensions (should be handled by padding)\n        landmarks = np.array([\n            [-100, -100],\n            [200, 200],\n            [150, -50],\n            [-50, 150],\n            [1000, 1000],\n        ], dtype=np.float32)\n        \n        face = _make_tracked_face(10, 10, 90, 90, label=\"hero\", landmarks=landmarks)\n        \n        # Should not crash even with extreme coordinates\n        result = enhancer.enhance_faces(frame, [face], frame_idx=0)\n        assert result.shape == frame.shape\n\n    def test_extreme_enhancement_models(self):\n        \"\"\"Test edge cases with different enhancement model names.\"\"\"\n        # Test with None model\n        cfg = _make_cfg(enable=True, model=None)\n        enhanced_cfg = PipelineConfig(\n            enable_face_enhancement=True,\n            enhancement_weight=0.7,\n            enhancement_model=\"gfpgan\",  # Model is None but gfpgan is detected\n            enhance_model_path=None,\n        )\n        enhancer = FaceEnhancer(enhanced_cfg)\n        # Should handle model=None gracefully\n\n    def test_timing_thresholds(self):\n        \"\"\"Test that enhancement meets minimum performance thresholds.\"\"\"\n        import time\n        \n        cfg = _make_cfg(enable=True)\n        enhancer = FaceEnhancer(cfg)\n        \n        frame = _frame(480, 480)\n        \n        # Measure enhancement time\n        start_time = time.time()\n        for _ in range(5):\n            enhancer.enhance(frame, frame_idx=0)\n        end_time = time.time()\n        \n        avg_time = (end_time - start_time) / 5\n        # Should complete within reasonable time (adjust threshold as needed)\n        assert avg_time < 0.5, f\"Enhancement took too long: {avg_time}s\"\n\n    def test_stabilization_extreme_values(self):\n        \"\"\"Test stabilizer with extreme temporal values.\"\"\"\n        cfg = _make_cfg(enable=True)\n        enhancer = FaceEnhancer(cfg)\n        \n        # Test with large frame_id differences\n        box = (10, 10, 100, 100)\n        result = enhancer._stabilize_enhancement_box(1, 1000, box, 500, 500)\n        assert len(result) == 4\n        \n        # Test with zero and negative alpha values\n        base = np.ones((32, 32, 3), dtype=np.float32) * 100.0\n        zero = np.zeros((32, 32, 3), dtype=np.float32)\n        \n        result1 = enhancer._stabilize_enhancement_residual(5, 0, base)\n        result2 = enhancer._stabilize_enhancement_residual(5, 1, zero)\n        \n        assert result1.shape == base.shape\n        assert result2.shape == zero.shape\n\n    def test_multiple_stabilization_states(self):\n        \"\"\"Test state management with multiple stabilization operations.\"\"\"\n        cfg = _make_cfg(enable=True)\n        enhancer = FaceEnhancer(cfg)\n        \n        # Add multiple states to internal trackers\n        enhancer._track_boxes[1] = (1, (10, 20, 100, 200))\n        enhancer._track_resids[2] = (2, np.ones((16, 16, 3), dtype=np.float32) * 50.0)\n        \n        # Test retrieving these states\n        result = enhancer._stabilize_enhancement_box(1, 2, (50, 50, 150, 150), 200, 200)\n        assert len(result) == 4\n\n    def test_extreme_enhancement_sizes(self):\n        \"\"\"Test enhancement with various crop sizes.\"\"\"\n        cfg = _make_cfg(enable=True)\n        enhancer = FaceEnhancer(cfg)\n        \n        # Test with different crop sizes\n        for crop_size in [(1, 1, 3, 3), (7, 7, 100, 100), (1000, 1000, 2000, 2000)]:\n            crop = _frame(*crop_size[2:])\n            result = enhancer.enhance_crop(crop)\n            assert result.shape == crop.shape\n\n\nif __name__ == \"__main__\":\n    import pytest\n    pytest.main([__file__, \"-v\"])"}}
</task_progress>
- [ ] Explore project structure around face_enhancer.py
- [x] Listed chararep directory contents  
- [x] Understood PipelineConfig structure from config.py
- [x] Understood pipeline.py use of FaceEnhancer (region of code, involved objects/data/steps)
- [x] Read face_enhancer.py to understand implementation
- [x] Listed test directory
- [x] Read test_chararep_face_enhancer.py to understand testing approach
- [x] Read config.py to understand enhancement_model configuration
- [x] Identified existing issues and improvements
- [x] Added new edge case tests</task_progress>
    </write_to_file>
]<|END_ACTION|>