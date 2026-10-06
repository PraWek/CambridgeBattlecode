"""Read per-team runtime errors and timeouts from local .replay26 files."""

from pathlib import Path


def varint(data, offset):
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise ValueError('truncated replay varint')
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
    raise ValueError('invalid replay varint')


def fields(data):
    offset = 0
    while offset < len(data):
        tag, offset = varint(data, offset)
        wire = tag & 7
        if wire == 0:
            value, offset = varint(data, offset)
        elif wire in (1, 2, 5):
            if wire == 2:
                size, offset = varint(data, offset)
            else:
                size = 8 if wire == 1 else 4
            if offset + size > len(data):
                raise ValueError('truncated replay field')
            value = data[offset:offset + size]
            offset += size
        else:
            raise ValueError(f'unsupported replay wire type: {wire}')
        yield tag >> 3, value


def health(path):
    teams = [{'timeouts': 0, 'errors': 0, 'error_samples': [], 'max_cpu_us': 0}
             for _ in range(2)]
    owners = {}
    for field, payload in fields(Path(path).read_bytes()):
        if field == 1:
            for map_field, entity in fields(payload):
                if map_field == 4:
                    entity = dict(fields(entity))
                    owners[entity.get(1, 0)] = entity.get(2, 0)
        elif field == 3:
            for turn_field, update in fields(payload):
                if turn_field != 1:
                    continue
                for event, value in fields(update):
                    if event == 1:
                        placed = dict(fields(value))
                        entity = dict(fields(placed[1]))
                        owners[entity.get(1, 0)] = entity.get(2, 0)
                    elif event == 9:
                        output = dict(fields(value))
                        owner = owners.get(output.get(1, 0))
                        if owner is None:
                            continue
                        stats = teams[owner]
                        stats['timeouts'] += bool(output.get(4, 0))
                        stats['max_cpu_us'] = max(stats['max_cpu_us'], output.get(3, 0))
                        text = output.get(2, b'').decode('utf8', errors='replace')
                        if any(token in text for token in ('Traceback (most recent call last)', 'Error:', 'Exception:')):
                            stats['errors'] += 1
                            if len(stats['error_samples']) < 3:
                                stats['error_samples'].append(text[-1500:])
    return teams
