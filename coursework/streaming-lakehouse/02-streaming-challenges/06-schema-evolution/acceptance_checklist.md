# Acceptance Checklist: Schema Evolution

Student ID: `demo000000`

## 6.1 Environment

- [ ] Paimon table is created successfully.
- [ ] Initial schema is defined correctly.
- [ ] Explain schema evolution and why it matters.
- [ ] Explain common schema changes in business systems.

## 6.2 Reproduce Schema Change Problem

- [ ] Multiple schema changes are executed, such as add field, delete field, and change type.
- [ ] Query failure or data loss is observed before compatible handling.
- [ ] Problem behavior is recorded.
- [ ] Explain why schema changes can cause problems.
- [ ] Screenshot `6-1`: error caused by schema change.

## 6.3 Implement Schema Evolution

- [ ] Paimon schema evolution support is configured.
- [ ] Backward-compatible schema changes are implemented.
- [ ] Default values are used for new fields.
- [ ] Explain Paimon schema evolution and compatible change types.

## 6.4 Verify Result

- [ ] Query works after schema change.
- [ ] Historical data can be read correctly.
- [ ] New data can be written correctly.
- [ ] Explain the verification process and result.
- [ ] Screenshot `6-2`: normal query result after schema evolution.
