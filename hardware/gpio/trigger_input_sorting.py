import gpiod
from gpiod.line import Direction, Value
import time
import threading
import queue

class TriggerSorting(threading.Thread):
    def __init__(self,trigger_sorting, stop_event, chip_path="/dev/gpiochip2",offset =0, internal=0.1):
        super().__init__()
        self.chip_path = chip_path
        self.offset = offset
        self.count = 0 
        self.chip = gpiod.Chip(chip_path)
        self.trigger_sorting = trigger_sorting
        self.internal = internal

        self.stop_event = stop_event
        

    
    def run(self):
        print("[TriggerSorting] Started")
        settings = gpiod.LineSettings()
        settings.direction = Direction.INPUT

        request = self.chip.request_lines(
            consumer = 'trigger-sorting',
            config = {self.offset: settings}
        )

        
        prev_signal = request.get_values([self.offset])[0]

        try:
            while not self.stop_event.is_set():
                cur_signal = request.get_values([self.offset])[0]
                
                # Down edge: 1 -> 0
                if prev_signal == Value.ACTIVE and cur_signal == Value.INACTIVE:
                    print('[TriggerSorting] trigger input')
                    #  push trigger
                    try:
                        self.trigger_sorting.put_nowait(1)
                    except queue.Full:
                        try:
                            self.trigger_sorting.get_nowait()
                            self.trigger_sorting.put_nowait(1)
                            
                        except queue.Empty:
                            pass

                    

                    prev_signal = cur_signal

                elif prev_signal == Value.INACTIVE and cur_signal == Value.ACTIVE:
                    prev_signal = cur_signal

                time.sleep(self.internal)


        except Exception as e:
            print(f"[TriggerSorting Error] {e}")

        finally:
            request.release()
            self.chip.close()
            
