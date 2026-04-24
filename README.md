# Cambridge Battlecode

## Quick start
```
pip install cambc
cambc starter
cambc run starter starter --watch
```

## Game rules

### Background

The year is 2076. A crystalline ore called **axionite** — a room-temperature superconductor — has been discovered on Titan, Saturn's largest moon. At least six corporations have deployed autonomous extraction fleets to Titan's surface.

Titan is lethal to humans: −179°C, a nitrogen-methane atmosphere, and a 76-minute communication delay to Earth. All operations are carried out by robots.

You write the software that controls your fleet: mining ore, refining axionite, and outcompeting the enemy.

### Objective

Collect resources and **destroy the enemy's core**.

To do this, you must find ore deposits, build harvesters, deliver resources back to the core using conveyors, and expand your territory.

### Win conditions

A game ends immediately if a core is destroyed or a team calls `resign()`. Otherwise, if both cores are still alive after **2000 rounds**, the winner is decided by tiebreakers in order:

<Steps>
  <Step title="Axionite delivered">
    Total refined axionite delivered to the core
  </Step>

  <Step title="Titanium delivered">
    Total titanium delivered to the core
  </Step>

  <Step title="Harvesters alive">
    Number of harvesters currently alive
  </Step>

  <Step title="Axionite stored">
    Total axionite currently stored
  </Step>

  <Step title="Titanium stored">
    Total titanium currently stored
  </Step>

  <Step title="Coinflip">
    Random tiebreaker
  </Step>
</Steps>

<Info>
  Axionite converted by the core with `c.convert(...)` is removed from the Ax
  collected stat and added to the Ti collected stat.
</Info>

### Map

The map is a rectangular grid between **20×20** and **50×50** inclusive. The top-left (northwest) corner is position `(0, 0)`.

The map is guaranteed to be **symmetric** by either reflection or rotation.

Walls prevent building anything on the tile they occupy. Ores are tiles on which harvesters may be built to mine resources.

### Units

Units are game entities which run an **independent instance** of the code that you submit. The core, builder bots, and turrets are units. **Harvesters are not units** — they operate automatically.

Each round, units take their turns **in the order they were spawned**. After all units have taken their turn, resources are distributed. See the reference tables for a quick comparison of all entity stats.

Each team can have at most **50 living units total**, including the core. In practice, that means a team can have at most **49 additional living units** at once. You can inspect the current count with `c.get_unit_count()`, and the cap is exposed as `GameConstants.MAX_TEAM_UNITS`.

### Vision and action radius

Units have a **vision radius** and an **action radius**.

* The **vision radius** is the area in which the unit can sense its environment.
* The **action radius** is the area in which the unit can perform actions, such as building, placing markers, or destroying buildings.

All units have an action radius of √2 (r² = 2), except the core, which has an action radius of √8 (r² = 8) measured from its centre.

Turrets also have an **attack range** which is different from their action radius — see each turret's page for details.

### Cooldowns

All units have **action** and **move** cooldowns which are non-negative integers that decrease by 1 at the end of each round. Actions and movement can only be performed when the respective cooldown is 0.

<Info>
  The move cooldown is only used by the builder bot — it is the only mobile unit.
</Info>

### Markers

All units may place one marker per round on a tile within action radius. This is separate from the action cooldown. You can overwrite an existing friendly marker, but not an enemy marker.

### Self-destruct

All units may self-destruct at any time. Builder bots do **not** deal damage when they self-destruct.

### Buildings

Buildings are game entities which are **immovable**. All entities are buildings except builder bots.

In particular, the core and turrets are considered **both a unit and a building**.

### Entity IDs

All entities (buildings and units) in the game have a **unique integer ID**. All Controller methods that deal with entities accept and return these IDs. Properties of an entity can be queried with getter functions — for example, `c.get_hp(id)`.

```python theme={"dark"}
# Get all nearby entities and check their type
for entity_id in c.get_nearby_entities():
    if c.get_entity_type(entity_id) == EntityType.HARVESTER:
        pos = c.get_position(entity_id)
```

<Info>
  The ID-based API was chosen for performance — constructing Python objects for every entity query would be too slow within the 2ms time limit.
</Info>

### Resource IDs

Each stored resource stack also has a **unique integer ID**. Use these IDs when you want to tell otherwise identical stacks apart in your logistics network.

You can query the stack currently stored in a conveyor, splitter, armoured conveyor, bridge, or foundry with `c.get_stored_resource_id(id)`. If the building is empty, it returns `None`.

```python theme={"dark"}
resource_type = c.get_stored_resource(conveyor_id)
resource_id = c.get_stored_resource_id(conveyor_id)
```

### Computation limit

Each unit gets **2ms of CPU time** per round. If your code exceeds this, execution is interrupted and `run()` is called fresh on the next round — **your bot does not resume where it left off**.

To absorb variance, each unit has an **extra time buffer** equal to 5% of the time limit. If a round takes longer than 2ms, the overage is deducted from the buffer. If a round takes less than 2ms, the savings are refunded to the buffer (capped at 5%).

Once a unit exhausts both its 2ms budget and its buffer in a single round, it is interrupted immediately.

Each bot process has a **1 GB memory limit**. Exceeding this will terminate the process.

Only Python standard library modules are available. External packages (e.g. `numpy`, `scipy`) cannot be imported — bots run in a sandboxed environment with no `pip install`.

<Warning>
  The local runner (`cambc run`) does **not** enforce time limits. Use `cambc match test` to test on the same AWS Graviton3 hardware that runs ladder matches.
</Warning>

### Debugging

* **stdout** via `print("msg")` is captured by the engine and saved to the replay. You can view each unit's output in the visualiser.
* **stderr** prints to the console in real time — use this for debugging during local runs.
* `c.draw_indicator_line(pos_a, pos_b, r, g, b)` and `c.draw_indicator_dot(pos, r, g, b)` draw debug overlays on the map, saved to the replay.
