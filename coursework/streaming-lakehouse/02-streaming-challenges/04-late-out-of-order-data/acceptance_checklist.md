# Acceptance Checklist: Late And Out-Of-Order Data

Student ID: `demo000000`

## 4.1 Environment

- [ ] Producer can generate late and out-of-order events.
- [ ] Flink Watermark strategy is configured.
- [ ] Explain late data, out-of-order data, and the role of Watermark.

## 4.2 Reproduce Problem

- [ ] Late and out-of-order events are sent.
- [ ] Late events are dropped or produce inaccurate results before optimization.
- [ ] Problem behavior is recorded.
- [ ] Explain why late data can make results inaccurate.
- [ ] Screenshot `4-1`: log showing dropped late data.

## 4.3 Implement Solution

- [ ] Suitable Watermark strategy is configured.
- [ ] Allowed lateness is configured.
- [ ] Side Output is used to handle late events.
- [ ] Explain the cooperation of Watermark, allowed lateness, and side output.

## 4.4 Verify Optimization

- [ ] Late data is handled correctly.
- [ ] Final results are accurate.
- [ ] Side Output captures late data correctly.
- [ ] Explain the late-data handling flow and result.
- [ ] Screenshot `4-2`: late data captured by side output.
- [ ] Screenshot `4-3`: final correct result.
