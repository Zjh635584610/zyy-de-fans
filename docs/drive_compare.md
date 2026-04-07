# Cityscape 与 CityscapeLite 驱动对比

当前对比已经基于以下修正重新测量：

- 使用 `qlabs_setup.py` 的默认 setup 起点
- 修正风险判断到车辆自身坐标系，不再用世界坐标 x 轴误判前方障碍
- 使用当前 `configs/control.yaml` 中的安全阈值

## 1. setup 逻辑结论

课题目录里的 [qlabs_setup.py](c:/Users/ZJH/Desktop/自控实验/实验程序/基于激光雷达的障碍物检测/qlabs_setup.py) 与参考例程的核心 spawn 逻辑是一致的：

- `QCar2` 采用 `location=[x*10 for x in initialPosition]`
- 通过 `spawn_id(..., waitForConfirmation=True)` 生成车辆
- 之后由 `QLabsRealTime().start_real_time_model(rtModel)` 启动实时模型

导致“初始化异常”的关键问题不是 `spawn` 写法，而是之前调用方没有按例程传入正确的初始位姿，并且风险判断使用了错误的世界坐标前向定义。现在这两个问题都已修正。

## 2. 最终对比结果

### Cityscape

- 结果文件：`docs/live_drive_summary_cityscape.json`
- 起点：`[0.0, 0.13024, -1.5707963267948966]`
- 采样帧数：`24`
- 平均检测目标数：`19.79`
- 告警帧数：`0`
- 紧急制动帧数：`0`
- 平均单帧处理时间：`0.0147462 s`
- 最大单帧处理时间：`0.0166828 s`
- 平均油门：`0.06634`
- 平均目标速度：`0.35`
- 平均实际速度：`0.27772`

### CityscapeLite

- 结果文件：`docs/live_drive_summary_cityscapelite.json`
- 起点：`[0.0, 0.13024, -1.5707963267948966]`
- 采样帧数：`24`
- 平均检测目标数：`16.17`
- 告警帧数：`0`
- 紧急制动帧数：`0`
- 平均单帧处理时间：`0.0145324 s`
- 最大单帧处理时间：`0.0179045 s`
- 平均油门：`0.06685`
- 平均目标速度：`0.35`
- 平均实际速度：`0.27774`

## 3. 结论

- 修正 setup 起点和风险判断后，车辆已经能从 setup 点稳定起步，不再“卡在地里”。
- `CityscapeLite` 相比 `Cityscape`，平均处理时间略低：
  - `Cityscape`: `14.75 ms`
  - `CityscapeLite`: `14.53 ms`
- 差异很小，不能说明 Lite 场景对时延有本质改善。
- 但 `CityscapeLite` 的检测目标数更少，场景更干净，更适合作为后续避障控制联调场景。

## 4. 建议

后续继续开发时：

- 保持 `CityscapeLite` 作为默认联调场景
- 保留 setup 起点不变
- 继续在此基础上接入最小转向避障，而不是再改初始化逻辑
