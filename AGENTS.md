# RC coordinate ownership

`TileCache` is the sole owner of `Position` objects in `bots/rc`.

- Allocate the complete coordinate pool only in `TileCache.__init__`.
- Never call `Position(...)` outside that pool and never call `pos.add(...)`.
- Canonicalize positions received from the Controller or marker payloads with
  `TileCache.canonicalize()`.
- Use `TileCache.position_at()`, `TileCache.neighbor()`, and
  `TileCache.offset()` for every derived coordinate.  Each returns the
  preallocated object (or `None` beyond the map boundary).
- Keep `tests/test_rc_position_ownership.py` passing.  It enforces these
  restrictions for all RC source files.
