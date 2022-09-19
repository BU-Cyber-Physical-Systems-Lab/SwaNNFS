from multiprocessing import Process, Queue
import time

def live_training(nn_queue, obs_queue):
    i = 0
    while True:
        i+=1
        print("got obs:", obs_queue.get())
        if i % 10 == 0:
            time.sleep(7)
            nn_queue.put("nn")

def clear_queue(q):
    while not q.empty():
        q.get()

def tranceive(nn_queue, obs_queue, circular_buffer_size=9):
    obs_dropped_count = 0
    i = 0
    prev_obs = None
    while True:
        while nn_queue.empty():
            time.sleep(0.5)
            i+=1
            if i % 30 == 0:
                i += 1
            checked_obs = (i, 1, 2)
            if prev_obs:
                if (prev_obs[0]+1) == checked_obs[0]:
                    obs_queue.put((prev_obs, checked_obs))
                    if obs_queue.qsize() > circular_buffer_size:
                        obs_queue.get()
                else:
                    obs_dropped_count += 1
                    print("obs dropped:", obs_dropped_count)
            prev_obs = checked_obs
        nn = nn_queue.get()
        print(nn)
        clear_queue(nn_queue)

if __name__ == '__main__':
    # ser = xbee.init()
    nn_queue = Queue()
    obs_queue = Queue()
    # time.sleep(3)

    Process(target=live_training, args=(nn_queue,obs_queue)).start()
    tranceive(nn_queue, obs_queue)
    #Thread(target=lambda: receiver.keep_receiving(ser)).start()

