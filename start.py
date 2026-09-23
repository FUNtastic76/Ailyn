import subprocess
import sys

print("Woking up Ailyn..")

try:

    sprite_process = subprocess.Popen([sys.executable, "sprite.py"])
    print("Visual interface started working.")

    core_process = subprocess.Popen([sys.executable, "tetris_arena.py"])
    print("Core started working.")

    # wait until two windows would be closed
    core_process.wait()
    sprite_process.wait()

except FileNotFoundError as e:
    print(f"{e.filename} file is not found")
except KeyboardInterrupt:
    print("\n  Stopping Ailyn...")
    core_process.terminate()
    sprite_process.terminate()

#binary detection -> digits -> tetris(?)
#убрать костыли и подобрать параметры
#улучшить архитектуру
'''
1. новая сетчатка(которая видить все одновременно, а не по отделности через цикл) --complete
2. моторная кара --completed for NOW
3. reinforcement learning -- completed
4. сама принимает решение --completed?
5. сон для Айлин чтобы лучшее запоминала -- completed
6. хоть какая то память чтобы понимала где ошиблась, где получилось --?
7. можеть смотреть на как я играю и если хочеть учится так --not yet
8. Эфферентная копия (Проприоцепция и ощущение своих действий) --completed
9. След синаптической памяти (Eligibility Trace / Проблема отложенной награды) --completed
10. Темпоральная динамика (Чувство времени) --completed
11. Фокус внимания (Ретикулярная формация) --?
12. Персистентное состояние между кадрами --completed
13. Мысленное симулирование (rollout) --completed
14. Механизм исследования (exploration) и использования (exploitation) --?
'''