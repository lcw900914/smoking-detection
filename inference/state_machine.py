"""階段順序狀態機:檢查 S1→S2→S3 序列合理性,輸出單週期分數。

- 在滑動視窗(預設 8 秒)內尋找 S1→S2→S3 依序出現
- S2 停留時長需 ≥ s2_min_dwell(預設 0.8 秒)
- 順序完整得 1.0;缺項或亂序打折

階段編號約定(與 data/dataset.py 一致):
    0 = S1 舉手、1 = S2 嘴部停留、2 = S3 放下、3 = background
"""
from collections import deque
from typing import Deque, List, Optional, Tuple

S1, S2, S3, BG = 0, 1, 2, 3


class StageStateMachine:
    """單一 track 的階段序列檢查器。

    每次時序頭推理後呼叫 `push(stage_id, timestamp)`,
    再以 `score()` 取得目前視窗的單週期狀態機分數。
    """

    def __init__(self, window_sec: float = 8.0, s2_min_dwell: float = 0.8):
        self.window_sec = window_sec
        self.s2_min_dwell = s2_min_dwell
        self._history: Deque[Tuple[float, int]] = deque()  # (時間戳, 階段)

    def push(self, stage_id: int, timestamp: float) -> None:
        """記錄一次階段判定(stage_id 為 argmax 後的類別)。"""
        self._history.append((timestamp, int(stage_id)))
        # 移出視窗外的舊紀錄
        while self._history and \
                timestamp - self._history[0][0] > self.window_sec:
            self._history.popleft()

    def _segments(self) -> List[Tuple[int, float, float]]:
        """把逐幀階段壓成連續段落 [(stage, t_start, t_end), ...]。"""
        segs: List[Tuple[int, float, float]] = []
        for t, s in self._history:
            if segs and segs[-1][0] == s:
                segs[-1] = (s, segs[-1][1], t)
            else:
                segs.append((s, t, t))
        return segs

    def score(self) -> float:
        """回傳目前視窗的狀態機分數 [0, 1]。

        評分規則:
        - S1→S2(dwell 足)→S3 依序出現:1.0
        - S2 dwell 足但缺 S1 或 S3 其中之一:0.6
        - S2 出現但 dwell 不足:0.3
        - 僅 S1 / S3(無 S2):0.15
        - 其他:0.0
        """
        segs = [(s, t0, t1) for s, t0, t1 in self._segments() if s != BG]
        if not segs:
            return 0.0

        stages_present = {s for s, _, _ in segs}
        s2_segs = [(t0, t1) for s, t0, t1 in segs if s == S2]
        s2_dwell_ok = any(t1 - t0 >= self.s2_min_dwell for t0, t1 in s2_segs)

        # 檢查是否存在依序的 S1 → S2(dwell 足)→ S3
        if s2_dwell_ok:
            for i, (s, _, _) in enumerate(segs):
                if s != S2:
                    continue
                t0, t1 = segs[i][1], segs[i][2]
                if t1 - t0 < self.s2_min_dwell:
                    continue
                has_s1_before = any(ss == S1 for ss, _, _ in segs[:i])
                has_s3_after = any(ss == S3 for ss, _, _ in segs[i + 1:])
                if has_s1_before and has_s3_after:
                    return 1.0
            # S2 dwell 足但缺 S1 或 S3
            if S1 in stages_present or S3 in stages_present:
                return 0.6
            return 0.6
        if S2 in stages_present:
            return 0.3
        if S1 in stages_present or S3 in stages_present:
            return 0.15
        return 0.0

    def reset(self) -> None:
        self._history.clear()


def cycle_score(sm_score: float, net_score: float,
                w_sm: float = 0.5, w_net: float = 0.5) -> float:
    """單週期分數 = w_sm × 狀態機分數 + w_net × 網路 cycle score。"""
    return w_sm * sm_score + w_net * net_score


class HandToMouthCounter:
    """手到嘴事件計數器:以「次數」決定警戒等級,而非單次動作觸發。

    以停留時長區分動作語意(抽菸吸一口的手到嘴有明確的時間窗):
        dwell < min_dwell        → 戴耳機/扶眼鏡/摸臉,不計
        min_dwell ≤ dwell ≤ max_dwell → 抽菸的一口,計一次事件
        dwell > max_dwell(一直舉著)  → 講電話,不計
        (進行中即可由 ongoing_dwell() 判斷講電話狀態,不必等放下)

    事件與上次事件間隔 ≥ min_gap 才計(避免同一口重複計數)。
    滾動視窗內:1 次 → 低(0.2)、2 次 → 中(0.5)、≥3 次 → 高(0.8)。
    """

    def __init__(self, window_sec: float = 90.0, min_dwell: float = 2.0,
                 max_dwell: Optional[float] = 5.0,
                 min_gap: float = 2.0, gap_tolerance: float = 0.5,
                 levels: Tuple[Tuple[int, float], ...] = ((1, 0.2), (2, 0.5),
                                                          (3, 0.8))):
        self.window_sec = window_sec
        self.min_dwell = min_dwell
        self.max_dwell = max_dwell            # None = 不設上限
        self.min_gap = min_gap
        # 骨架偵測會斷斷續續,S2 中斷 ≤ gap_tolerance 秒視為同一次停留
        self.gap_tolerance = gap_tolerance
        self.levels = sorted(levels)          # [(次數門檻, 分數), ...]
        self._events: Deque[float] = deque()  # 事件完成時間
        self._s2_start = None                 # 進行中的 S2 起點
        self._s2_last = None                  # 最後一次看到 S2 的時間
        self._last_event = float("-inf")

    def update(self, stage_id: int,
               timestamp: float) -> Optional[Tuple[float, bool, str]]:
        """推入一次階段判定;S2 中斷超過容忍值時結算是否構成事件。

        Returns:
            一段停留結算時回傳 (dwell 秒數, 是否計入, 原因),
            供呼叫端顯示「為什麼沒計數」;其餘時刻回傳 None。
        """
        result = None
        if stage_id == S2:
            if self._s2_start is None:
                self._s2_start = timestamp
            self._s2_last = timestamp
        elif self._s2_start is not None and \
                timestamp - self._s2_last > self.gap_tolerance:
            dwell = self._s2_last - self._s2_start
            if dwell < self.min_dwell:
                result = (dwell, False, "太短(扶眼鏡/摸臉)")
            elif self.max_dwell is not None and dwell > self.max_dwell:
                result = (dwell, False, "太長(講電話)")
            elif self._s2_last - self._last_event < self.min_gap:
                result = (dwell, False, "與上次事件間隔不足")
            else:
                self._events.append(self._s2_last)
                self._last_event = self._s2_last
                result = (dwell, True, f"計入第 {len(self._events)} 次")
            self._s2_start = None
            self._s2_last = None
        # 移出視窗外的舊事件
        while self._events and \
                timestamp - self._events[0] > self.window_sec:
            self._events.popleft()
        return result

    def ongoing_dwell(self, timestamp: float) -> float:
        """目前進行中的 S2 停留已持續秒數(無進行中停留回傳 0)。

        超過 max_dwell 即可即時判定「講電話」姿態,不必等手放下。
        """
        if self._s2_start is None or self._s2_last is None:
            return 0.0
        if timestamp - self._s2_last > self.gap_tolerance:
            return 0.0  # 已中斷,待下次 update 結算
        return timestamp - self._s2_start

    def count(self) -> int:
        """目前視窗內的事件次數。"""
        return len(self._events)

    def score(self) -> float:
        """次數 → 警戒分數(取達標的最高等級)。"""
        n = self.count()
        s = 0.0
        for need, val in self.levels:
            if n >= need:
                s = val
        return s

    def reset(self) -> None:
        self._events.clear()
        self._s2_start = None
        self._s2_last = None
        self._last_event = float("-inf")


class MovementGate:
    """移動排除(可開關):視窗內累積移動過大 → 不視為抽菸。

    移動量以「人物自身身高」為單位(路徑長 ÷ 平均框高),
    與攝影機距離無關:走動中的人手部擺動易誤判,且抽菸行為
    本質上是定點或慢移動的。
    """

    def __init__(self, max_heights: float = 3.0, window_sec: float = 10.0):
        """
        Args:
            max_heights: 視窗內累積路徑 ≥ 此倍數×人物身高 → 排除
            window_sec: 滾動視窗秒數
        """
        self.max_heights = max_heights
        self.window_sec = window_sec
        self._hist: Deque[Tuple[float, float, float, float]] = deque()
        # (t, cx, cy, 框高)

    def update(self, timestamp: float, bbox) -> bool:
        """推入一幀框,回傳目前是否「移動過大」。"""
        x1, y1, x2, y2 = [float(v) for v in bbox]
        self._hist.append((timestamp, (x1 + x2) / 2, (y1 + y2) / 2,
                           max(1.0, y2 - y1)))
        while self._hist and \
                timestamp - self._hist[0][0] > self.window_sec:
            self._hist.popleft()
        if len(self._hist) < 2:
            return False

        path = 0.0
        pts = list(self._hist)
        for (_, x0, y0, _), (_, x1_, y1_, _) in zip(pts, pts[1:]):
            path += ((x1_ - x0) ** 2 + (y1_ - y0) ** 2) ** 0.5
        mean_h = sum(h for _, _, _, h in pts) / len(pts)
        return path / mean_h >= self.max_heights

    def reset(self) -> None:
        self._hist.clear()


class PresenceClassifier:
    """在場型態分類:經過 / 徘徊 / 等待。

    純幾何,從框的軌跡直接算 —— 這幾件事本來就是確定的,交給模型學
    只會把「一個可以隨時調的門檻」變成「要重訓才能改的權重」。

    量測一律以人物自身身高(框高)為單位,與攝影機距離無關:
        累積路徑  = 中心逐幀位移總和 ÷ 平均框高   (走了多少)
        位移範圍  = 起訖最大距離 ÷ 平均框高       (走多遠,來回會抵銷)
        速度      = 累積路徑 ÷ 經過秒數           (身高/秒)

    判定:
        在場短 + 有移動            → 經過(速度超過 run_speed 記為跑步)
        在場久 + 累積路徑大        → 徘徊(一直在動,但沒離開)
        在場久 + 累積路徑小        → 等待(待著不太動)
        其餘(剛出現、還看不準)    → unknown

    「累積路徑」與「位移範圍」分開看是關鍵:在原地來回踱步的人位移範圍
    很小,但累積路徑很大 —— 這正是徘徊,只看位移會誤判成等待。

    限制:影像空間的量測,朝鏡頭正面走來的人位移看起來很小,可能被判成
    等待。要根治得靠地面標定,目前先以側向/俯視鏡位為前提。
    """

    PASSING = "passing"      # 經過(走)
    RUNNING = "running"      # 經過(跑)
    WANDERING = "wandering"  # 徘徊
    WAITING = "waiting"      # 等待
    UNKNOWN = "unknown"      # 在場時間還不夠判斷

    def __init__(self, window_sec: float = 60.0, short_stay: float = 8.0,
                 long_stay: float = 20.0, pass_path: float = 1.0,
                 wander_path: float = 3.0, run_speed: float = 1.5):
        """
        Args:
            window_sec: 軌跡回看視窗秒數(在場時長仍以首次出現起算)
            short_stay: 在場短於此秒數且有移動 → 經過
            long_stay: 在場超過此秒數才判徘徊/等待
            pass_path: 判「經過」所需的最小累積路徑(倍數×身高)
            wander_path: 累積路徑超過此值(倍數×身高)→ 徘徊,否則等待
            run_speed: 平均速度超過此值(身高/秒)→ 記為跑步
        """
        self.window_sec = window_sec
        self.short_stay = short_stay
        self.long_stay = long_stay
        self.pass_path = pass_path
        self.wander_path = wander_path
        self.run_speed = run_speed
        self._first_ts: Optional[float] = None
        self._hist: Deque[Tuple[float, float, float, float]] = deque()
        # (t, cx, cy, 框高)

    def update(self, timestamp: float, bbox) -> str:
        """推入一幀框,回傳目前的在場型態。"""
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if self._first_ts is None:
            self._first_ts = timestamp
        self._hist.append((timestamp, (x1 + x2) / 2, (y1 + y2) / 2,
                           max(1.0, y2 - y1)))
        while self._hist and timestamp - self._hist[0][0] > self.window_sec:
            self._hist.popleft()

        stay = timestamp - self._first_ts
        path, _span, speed = self._metrics()

        if stay >= self.long_stay:
            return (self.WANDERING if path >= self.wander_path
                    else self.WAITING)
        if stay >= self.short_stay:
            return self.UNKNOWN          # 中間地帶:還不夠判,先不表態
        if path >= self.pass_path:
            return self.RUNNING if speed >= self.run_speed else self.PASSING
        return self.UNKNOWN

    def _metrics(self):
        """回傳 (累積路徑, 位移範圍, 速度),單位皆為身高倍數。"""
        if len(self._hist) < 2:
            return 0.0, 0.0, 0.0
        pts = list(self._hist)
        path = 0.0
        for (_, x0, y0, _), (_, x1, y1, _) in zip(pts, pts[1:]):
            path += ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        mean_h = sum(h for _, _, _, h in pts) / len(pts)
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        span = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
        dt = max(1e-6, pts[-1][0] - pts[0][0])
        return path / mean_h, span / mean_h, (path / mean_h) / dt

    def stats(self):
        """診斷用:(在場秒數, 累積路徑, 位移範圍, 速度)。"""
        path, span, speed = self._metrics()
        stay = (0.0 if self._first_ts is None
                else self._hist[-1][0] - self._first_ts) if self._hist else 0.0
        return stay, path, span, speed

    def reset(self) -> None:
        self._hist.clear()
        self._first_ts = None


class LoiterDetector:
    """逗留偵測:長時間在場 + 手部關鍵點幾乎不可見 + 位移小 → 逗留警告。

    針對背對鏡頭的漏檢情境:骨架看不到手無法確認抽菸,
    但「一直待著且看不到手」本身就值得升級為警告,不讓 case 靜默消失。
    """

    def __init__(self, min_duration: float = 20.0, move_ratio: float = 0.6,
                 wrist_vis_max: float = 0.1):
        """
        Args:
            min_duration: 觀察視窗秒數(在場需達此時長)
            move_ratio: 視窗內中心位移 < 此比例×平均框對角線 視為未大幅移動
            wrist_vis_max: 手腕可見幀比例 ≤ 此值 視為「看不到手」
        """
        self.min_duration = min_duration
        self.move_ratio = move_ratio
        self.wrist_vis_max = wrist_vis_max
        self._hist: Deque[Tuple[float, float, float, float, bool]] = deque()
        # (t, cx, cy, 對角線, 手腕可見)

    def update(self, timestamp: float, bbox, wrist_visible: bool) -> bool:
        """推入一幀觀測,回傳目前是否構成逗留警告。"""
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        diag = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        self._hist.append((timestamp, cx, cy, diag, bool(wrist_visible)))
        while self._hist and \
                timestamp - self._hist[0][0] > self.min_duration:
            self._hist.popleft()

        if not self._hist or \
                timestamp - self._hist[0][0] < self.min_duration * 0.95:
            return False  # 在場時間不足

        xs = [h[1] for h in self._hist]
        ys = [h[2] for h in self._hist]
        mean_diag = sum(h[3] for h in self._hist) / len(self._hist)
        disp = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
        vis_frac = sum(h[4] for h in self._hist) / len(self._hist)

        return (disp < self.move_ratio * mean_diag
                and vis_frac <= self.wrist_vis_max)

    def reset(self) -> None:
        self._hist.clear()
