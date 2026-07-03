# 3D CT 재구성 도구 (3D CT Reconstruction Tool) - 상세 코드 설명서

이 문서는 스탠포드 의과대학(Stanford Medicine) 등 임상 환경에서 얻어지는 복잡한 DICOM CT 데이터를 분석하고, 뼈(Bone)와 같은 특정 조직을 실시간 3D로 재구성하는 파이썬 애플리케이션의 핵심 동작 원리를 매우 자세히 설명합니다.

---

## 1. 프로젝트 구조 (Project Architecture)

이 프로젝트는 크게 **데이터 처리부(`dicom_utils.py`)**와 **UI 및 3D 렌더링부(`main.py`)** 두 가지 핵심 모듈로 나뉘어 있습니다.

* **`dicom_utils.py`**: 하드디스크의 원본 `.dcm` 파일들을 읽어 들여 물리적인 3D 배열(Array)과 공간 정보(Spacing)로 변환하는 백엔드 엔진입니다.
* **`main.py`**: PyQt5를 기반으로 한 그래픽 사용자 인터페이스(GUI) 컨트롤러이자, PyVista(VTK)를 활용해 3D 렌더링을 수행하는 메인 실행 파일입니다.

---

## 2. `dicom_utils.py` 핵심 기능 분석

DICOM 데이터는 단순히 그림 파일이 아니라, 환자의 메타데이터와 3D 공간상의 좌표계를 포함하는 복잡한 포맷입니다. 이를 안전하고 정확하게 파싱하기 위한 여러 장치들이 구현되어 있습니다.

### 2.1. 시리즈(Series) 분리 및 필터링
하나의 CT 촬영 폴더에는 메인 3D 볼륨(Axial)뿐만 아니라 2D 관찰용(Coronal, Sagittal) 재구성 이미지나 방사선 피폭량 리포트(Dose Report)가 섞여 있습니다.
* **`load_scan(path)`**: 이 함수는 폴더 내의 모든 DICOM 파일을 읽은 뒤, 스카우트(Scout/Localizer) 이미지는 따로 빼내고, 나머지 이미지들을 `SeriesInstanceUID` 기준으로 그룹화합니다.
* **`get_series_info(series_dict)`**: 그룹화된 시리즈들의 장수(Slice count)와 설명(Description)을 UI에 표시하기 위해 추출하는 함수입니다. 이를 통해 사용자는 'Mako', 'Axial' 등 진짜 3D 원본 데이터만을 정확히 선택할 수 있습니다.

### 2.2. 강건한 슬라이스 정렬 (Robust Sorting)
* **`get_slice_position()`**: DICOM 슬라이스들은 파일명 순서대로 정렬되어 있지 않은 경우가 많습니다. 이 함수는 DICOM의 `ImageOrientationPatient`(환자 기준 X,Y 벡터)를 외적(Cross product)하여 법선(Normal) 벡터(Z축)를 구한 뒤, `ImagePositionPatient`(공간상 위치)를 내적(Dot product)하여 **수학적으로 완벽한 Z축 물리적 좌표**를 계산합니다. 이 좌표를 기준으로 슬라이스들을 위에서 아래로 정확히 정렬합니다.

### 2.3. 물리적 픽셀 간격(Spacing) 계산
* **슬라이스 두께 (Z축)**: 슬라이스 간의 Z축 간격을 단순히 헤더에서 읽어오는 대신, 인접 슬라이스 간의 실제 차이값(np.diff)들의 **중앙값(Median)**을 구합니다. 만약 간격의 표준편차(StdDev)가 너무 크면 비균일(Non-uniform) 간격 경고를 띄워 3D 비율이 왜곡될 수 있음을 사용자에게 알립니다.
* **픽셀 크기 (X, Y축)**: `PixelSpacing` 메타데이터에서 가로/세로 길이를 추출하여 (z, y, x) 튜플 형태의 물리적 간격을 완성합니다.

### 2.4. Hounsfield Units (HU) 변환
* **`get_pixels_hu(slices)`**: CT 스캐너가 출력하는 원본 픽셀 값은 기계마다 다릅니다. 이를 공통된 물리학적 밀도 단위인 Hounsfield Unit(HU)으로 변환합니다. `RescaleIntercept`와 `RescaleSlope`를 곱하고 더하는 선형 변환을 거칩니다. 이 과정에서 뼈는 보통 +300 ~ +3000 HU의 값을 가지게 됩니다.

---

## 3. `main.py` 핵심 기능 분석

사용자와 상호작용하고, 3D 렌더링 파이프라인을 구축하는 메인 컨트롤러입니다.

### 3.1. 고속 데이터 로딩 및 `v2` 캐싱 시스템
CT 데이터(수백~수천 장의 DICOM)를 읽고 HU로 변환하는 작업은 수 초~수십 초가 걸릴 수 있습니다.
* **`_cache_v2.npz`**: 환자를 처음 로드할 때 연산된 3D 배열(image_hu), 물리적 간격(spacing), 스카우트 이미지들, 환자 메타데이터(이름, 날짜, 장비)를 모두 Numpy의 `.npz` 무손실 압축 포맷으로 디스크에 저장합니다.
* 두 번째 로딩부터는 원본 DICOM을 전혀 읽지 않고 이 캐시 파일만 메모리에 통째로 올려, **수십 배 빠른 속도(거의 즉시 로딩)**를 자랑합니다. 이전 버전의 버그를 우회하기 위해 `v2` 명명 규칙을 적용했습니다.

### 3.2. 3D 메쉬 생성 파이프라인 (Contour Isosurface)
사용자가 지정한 Min/Max 밀도(HU) 범위의 조직만 3D로 뽑아내는 가장 핵심적인 로직입니다. 과거의 투박한 큐브(Voxel) 방식에서 수학적으로 완벽한 곡면 추출 방식으로 진화했습니다.
* **`update_base_mesh()`**: 
  1. 원본 3D 데이터는 보호하고, 복사본을 만들어 **Max 수치(예: 3000)를 초과하는 밀도를 마스킹(제거)**합니다.
  2. PyVista의 `contour()` 함수를 이용해 **Min 수치(예: 300)와 일치하는 완벽한 등위면(Isosurface)**을 뽑아냅니다. 마치 등고선을 그리듯 매끄러운 3D 표면이 도출되며, 예전처럼 레고 블록 형태의 깨진 폴리곤(Seam artifact)이 전혀 발생하지 않습니다.

### 3.3. 표면 스무딩 (Surface Smoothing)
추출된 뼈 표면의 자잘한 노이즈를 제거하여 임상 진단 및 3D 프린팅에 적합하게 만듭니다.
* **`mesh.clean()`**: 중복된 점(Coincident vertices)들을 하나로 합쳐 위상학적 오류를 제거합니다.
* **Laplacian (라플라시안)**: 꼭짓점을 주변 점들의 평균 위치로 당기는 가장 기본적인 스무딩 기법입니다.
* **Windowed Sinc / Taubin (토빈)**: 뼈의 날카로운 모서리(Feature)나 전체 부피(Volume)는 잃지 않으면서(수축 방지), 노이즈만 제거하는 고급 의료용 스무딩 기법입니다. (코드에서는 `smooth_taubin` 적용)

### 3.4. 실시간 3D 인터랙션 (Interactive 3D Widgets)
사용자가 3D 볼륨을 자유롭게 탐색할 수 있도록 다양한 VTK 위젯을 제공합니다.
* **3D Box Cropping (`on_crop_toggled`)**: 체크박스를 켜면 메쉬 주변에 육면체 상자가 생깁니다. 상자의 면을 마우스로 밀고 당기면, 실시간으로 메쉬가 잘려 나가는 클리핑(Clipping) 연산을 수행합니다.
* **Hide Crop Handles (`on_hide_handles_toggled`)**: 크롭 상자의 조절점(Handle)이 시야를 가리지 않도록, 상자 형태는 유지한 채 인터페이스 선들만 순간적으로 숨기는 기능입니다. 이를 위해 현재 Box의 공간 정보(Bounds)를 메모리에 기억해두는 로직이 구현되어 있습니다.

### 3.5. 다중 윈도우 UI 시스템 (Multi-Window UI)
* **Scout Viewer**: 환자의 촬영 위치를 확인하는 2D 맵인 스카우트 뷰어를 메인 화면에서 독립시켰습니다. `< Prev`, `Next >` 버튼을 통해 정면(AP)/측면(Lateral) 스카우트 이미지들을 넘겨볼 수 있으며, Matplotlib의 네비게이션 툴바를 통해 자유로운 확대/축소가 가능합니다.
* **Patient Info Viewer**: 캐시에 저장된 DICOM 메타데이터를 HTML 형식의 깔끔한 텍스트로 보여주는 별도의 윈도우입니다. 환자의 정보가 원본과 일치하는지, 해상도(mm)가 얼마인지 한눈에 검증할 수 있습니다.

---

## 4. 요약 및 향후 확장성

이 도구는 **1) 완벽한 원본 DICOM 보존**, **2) 사용자가 직접 제어하는 시리즈 선택**, **3) 매끄러운 Contour 3D 재구성**, **4) 고성능 압축 캐싱**을 달성했습니다. 

현재 구조는 데이터 로딩(IO) ➔ 전처리(Pre-processing) ➔ 3D 생성(Mesh generation) ➔ 렌더링(Rendering)이 명확히 분리되어 있어, 차후 특정 뼈 부위를 색칠하거나(Transfer Function) `.STL` 파일로 저장(Export)하는 기능을 붙이기에 매우 이상적인 구조를 갖추고 있습니다.
