# 원본 코드 위치(실행해야 하는 코드 위치) : /home/rokey/isaacsim/standalone_examples/tutorials/load_map.py

from isaacsim import SimulationApp

# 1. Isaac Sim 시뮬레이션 앱 초기화 (GUI를 보기 위해 headless를 False로 설정)
simulation_app = SimulationApp({"headless": False})

import omni.usd
from omni.isaac.core import World

# 사용자 지정 USD 파일 경로
usd_path = "/home/rokey/cobot3_ws/01_digital_twin_map/smcnd_factory_v4.usd"

# 2. 스테이지 열기 (파일 로드)
# open_stage는 기존 스테이지를 닫고 지정된 경로의 USD를 새 스테이지로 설정함
omni.usd.get_context().open_stage(usd_path)

# 3. 시뮬레이션 월드 객체 생성
# 스테이지 유닛(단위)을 미터(1.0)로 설정하여 물리 엔진 초기화
world = World(stage_units_in_meters=1.0)
world.reset()

# 4. 시뮬레이션 메인 루프
while simulation_app.is_running():
    # 물리 연산 및 렌더링 업데이트
    world.step(render=True)

# 앱 종료
simulation_app.close()