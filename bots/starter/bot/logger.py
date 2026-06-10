import os
from cambc import Controller, Direction, EntityType, Position

# Глобальный флаг для управления логированием
DEBUG = True
LOG_FILE = "log.txt"


class BotLogger:
    def __init__(self):
        self.log_file = None
        if DEBUG:
            # Очищаем файл логов при первом создании экземпляра (обычно, при старте Ядра)
            # В Cambridge Battlecode каждый юнит имеет свой процесс,
            # но мы хотим, чтобы log.txt был общим.
            # Для простоты, первый запущенный процесс (Ядро) будет очищать файл.
            # В реальных условиях с множеством процессов может потребоваться более сложная
            # координация, но для Battlecode этот подход приемлем.
            if os.path.exists(LOG_FILE):
                try:
                    os.remove(LOG_FILE)
                except OSError:
                    # Если файл уже используется другим процессом, просто продолжаем.
                    # Это может произойти, если несколько ботов стартуют почти одновременно.
                    pass
            self.log_file = open(LOG_FILE, 'a', buffering=1)  # Режим добавления, построчное кэширование

    def _format_log_message(self, ct: Controller, message: str) -> str:
        if not DEBUG:
            return ""

        round_num = ct.get_current_round()
        unit_id = ct.get_id()
        unit_type = ct.get_entity_type().name
        unit_pos = ct.get_position()
        return f"[Round {round_num}] [{unit_type} #{unit_id}] ({unit_pos.x}, {unit_pos.y}) -> {message}"

    def log_core_stats(self, ct: Controller):
        if not DEBUG:
            return
        titanium, axionite = ct.get_global_resources()
        scale_percent = ct.get_scale_percent()
        unit_count = ct.get_unit_count()
        msg = f"Resources: Ti={titanium}, Ax={axionite}, Scale={scale_percent:.1f}%, Units={unit_count}"
        self.log_file.write(self._format_log_message(ct, msg) + "\n")

    def log_move(self, ct: Controller, from_pos: Position, to_pos: Position):
        if not DEBUG:
            return
        msg = f"Moved from ({from_pos.x}, {from_pos.y}) to ({to_pos.x}, {to_pos.y})"
        self.log_file.write(self._format_log_message(ct, msg) + "\n")

    def log_build(self, ct: Controller, structure_type: EntityType, pos: Position, direction: Direction = None):
        if not DEBUG:
            return
        if direction:
            msg = f"Built {structure_type.name} pointing {direction.name.upper()} at ({pos.x}, {pos.y})"
        else:
            msg = f"Built {structure_type.name} at ({pos.x}, {pos.y})"
        self.log_file.write(self._format_log_message(ct, msg) + "\n")

    def log_info(self, ct: Controller, message: str):
        if not DEBUG:
            return
        self.log_file.write(self._format_log_message(ct, message) + "\n")

    def __del__(self):
        if self.log_file and not self.log_file.closed:
            self.log_file.close()
