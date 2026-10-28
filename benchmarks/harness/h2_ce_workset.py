"""Deterministic CE-only correctness inputs, not a representative speed workload."""
from __future__ import annotations

import copy


def repeat_row(row: dict, count: int) -> dict:
    if type(count) is not int or not 1 <= count <= 8:
        raise ValueError('Repetition must be an integer in [1,8]')
    if set(row) != {'input_ids', 'attention_mask', 'labels'}:
        raise ValueError('Unexpected source fields')
    ids, masks, labels = (row[k] for k in ('input_ids', 'attention_mask', 'labels'))
    if (not ids or len(ids) != len(masks) or len(ids) != len(labels)
            or any(type(x) is not int or x < 0 for x in ids)
            or any(type(x) is not int or x != 1 for x in masks)
            or labels[0] != -100
            or any(type(x) is not int or x not in (-100, ids[i])
                   for i, x in enumerate(labels))
            or all(x == -100 for x in labels)):
        raise ValueError('Expected complete unpadded, masked source row')
    return {key: value * count for key, value in row.items()}


def collate(rows: list[dict], pad_id: int) -> dict:
    if not rows or type(pad_id) is not int or pad_id < 0:
        raise ValueError('Nonempty rows and explicit pad ID required')
    checked = [repeat_row(row, 1) for row in rows]
    width = max(len(row['input_ids']) for row in checked)
    return {key: [row[key] + [pad] * (width - len(row[key])) for row in checked]
            for key, pad in [('input_ids', pad_id), ('attention_mask', 0), ('labels', -100)]}


def build_workset(source: list[dict]) -> dict:
    if len(source) != 128:
        raise ValueError('Expected all 128 frozen source rows')
    probes = [collate([repeat_row(source[index], count)], 128009)
              for index, count in [(0, 1), (10, 2), (63, 4)]]
    probes.append(collate([repeat_row(source[126], 1), repeat_row(source[127], 8)], 128009))
    order = [0, 1, 9, 10, 11, 31, 32, 63, 64, 65, 98, 99, 100, 101, 126, 127]
    return {'schema': 'h2-ce-workset-v1', 'probes': probes,
            'training_row_indices': order,
            'training_batches': [collate([copy.deepcopy(source[index])], 128009)
                                 for index in order]}
