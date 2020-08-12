from multiprocessing import Process, Queue
from sbir_util.batch_manager import MemoryBlockManager
from image_proc import Transformer
from sample_util import *


class SamplingLayer(object):
    def setup(self, sketch_dir, mean, hard_ratio, batch_size):
        """Setup the SamplingLayer."""
        self.create_sample_fetcher(sketch_dir, mean, hard_ratio, batch_size)

    def create_sample_fetcher(self, sketch_dir, mean, hard_ratio, batch_size):
        self._blob_queue = Queue(10)
        self._prefetch_process = SamplingDataFetcher(self._blob_queue, sketch_dir, mean, hard_ratio, batch_size)
        self._prefetch_process.start()
        def cleanup():
            print ('Terminating BlobFetcher')
            self._prefetch_process.terminate()
            self._prefetch_process.join()
        import atexit
        atexit.register(cleanup)

    def get_next_batch(self):
        return self._blob_queue.get()


class SamplingDataFetcher(Process):
    def __init__(self, queue, sketch_dir,  mean, hard_ratio, batch_size):
        """Setup the StrokeSamplingDataLayer."""
        super(SamplingDataFetcher, self).__init__()
        #        mean = mean
        self._queue = queue
        self.sketch_transformer = Transformer(225, 1, mean)
        self.sketch_dir = sketch_dir
        self.sketch_bm = MemoryBlockManager(sketch_dir)
        self.hard_ratio = hard_ratio
        self.mini_batchsize = batch_size

    def get_next_batch(self):
        sketch_batch = []
        # sampling
        sketch_inds = self.sketch_bm.pop_batch_inds_circular(self.mini_batchsize)
        # fetch data
        for (sketch_id) in zip(sketch_inds):
            sketch_batch.append(self.sketch_bm.get_sample(sketch_id).reshape((256, 256, 1)))
        # apply transform
        sketch_batch = self.sketch_transformer.transform_all(sketch_batch).astype(np.uint8)
        self._queue.put(sketch_batch)

    def run(self):
        print ('SamplingDataFetcher started')
        while True:
            self.get_next_batch()


if __name__ == '__main__':#hard ratio参数似乎没什么卵用
    print ('SamplingDataFetcher started')
    sketch_dir = 'data/shoes_new/sketch_train.mat'
    mean = 250.42
    hard_ratio = 0.75
    batch_size = 8
    sampler = SamplingLayer()
    sampler.setup(sketch_dir, mean, hard_ratio, batch_size)
    while True:
        sampler.get_next_batch()