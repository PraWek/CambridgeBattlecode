from cambc import Position, Environment, Direction
from bot.constants import ORTHOGONAL_DIRECTIONS


def compute_steiner_network(core_pos: Position, known_env: dict, radius: int = 10) -> tuple[
    dict[Position, Direction], set[Position]]:
    """
    Приближенно вычисляет направленное дерево Штейнера, соединяющее
    известные залежи титана в радиусе `radius` с Ядром.

    Возвращает:
        conveyors: dict[Position, Direction] - карта конвейеров и их направлений (к ядру).
        network_ores: set[Position] - список подключенных к сети руд.
    """
    ores = []
    for pos, env in known_env.items():
        if env == Environment.ORE_TITANIUM:
            # Chebyshev distance для радиуса
            if max(abs(pos.x - core_pos.x), abs(pos.y - core_pos.y)) <= radius:
                ores.append(pos)

    tree_nodes = {core_pos}
    conveyors = {}
    network_ores = set()

    unconnected_ores = set(ores)

    # Жадное добавление руды в дерево
    while unconnected_ores:
        queue = list(tree_nodes)
        came_from = {}
        found_ore = None

        # Multi-source BFS от текущих узлов дерева
        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1

            if current in unconnected_ores:
                found_ore = current
                break

            for d in ORTHOGONAL_DIRECTIONS:
                nxt = current.add(d)

                env = known_env.get(nxt, Environment.EMPTY)
                # Избегаем стен и другой руды (мы не можем строить конвейеры на аксионите)
                if env == Environment.WALL or env == Environment.ORE_AXIONITE:
                    continue

                if nxt not in came_from and nxt not in tree_nodes:
                    came_from[nxt] = current
                    queue.append(nxt)

        if not found_ore:
            break  # Оставшиеся руды недостижимы

        curr = found_ore
        network_ores.add(found_ore)

        # Обратный ход по пути: добавляем ветку в дерево Штейнера
        while curr not in tree_nodes:
            nxt_node = came_from[curr]
            # Направление конвейера - от текущей клетки к следующему узлу (к ядру)
            d = curr.direction_to(nxt_node)

            if curr != found_ore:
                conveyors[curr] = d

            tree_nodes.add(curr)
            curr = nxt_node

        unconnected_ores.remove(found_ore)

    return conveyors, network_ores