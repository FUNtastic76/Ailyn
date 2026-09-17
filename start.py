import subprocess
import sys

print("Woking up Ailyn..")

try:

    sprite_process = subprocess.Popen([sys.executable, "sprite.py"])
    print("Visual interface started working.")

    core_process = subprocess.Popen([sys.executable, "D:\AI\PythonProject\Ailyn_Brain\Ailyn_central_system.py"])
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
1. новая сетчатка(которая видить все одновременно, а не по отделности через цикл)
2. моторная кара
3. самообучение без учителя (reinforcement learning)
4. сама принимает решение
5. сон для Айлин чтобы лучшее запоминала
6. хоть какая то память чтобы понимала где ошиблась, где получилось
7. можеть смотреть на как я играю и если хочеть учится так
8. Эфферентная копия (Проприоцепция и ощущение своих действий)
9. След синаптической памяти (Eligibility Trace / Проблема отложенной награды)
10. Темпоральная динамика (Чувство времени)
11. Фокус внимания (Ретикулярная формация)
12. Персистентное состояние между кадрами
13. Мысленное симулирование (rollout)
14. Механизм исследования (exploration) и использования (exploitation)
'''