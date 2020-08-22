import tensorflow as tf

filename_queue = tf.train.string_input_producer(['data/t.csv'], num_epochs=3)
reader = tf.TextLineReader()
key, value = reader.read_up_to(filename_queue, num_records=3)
batch_values = tf.train.shuffle_batch([value], batch_size=3, capacity=64, min_after_dequeue=2, enqueue_many=True)

record_defaults = [[0]] * 5
col1, col2, col3, col4, col5 = tf.decode_csv(batch_values, record_defaults=record_defaults, field_delim=',')
features = tf.stack([col1, col2, col3, col4], axis=1)
labels = tf.reshape(tf.cast(col5, tf.int32), [3])

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    sess.run(tf.initialize_local_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        step = 0
        while not coord.should_stop():
            step += 1
            print('step = {}'.format(step))
            example, label = sess.run([features, labels])
            print(example)
            print(label)
    except tf.errors.OutOfRangeError:
        print('training for 1 epochs, {} steps'.format(step))
    finally:
        coord.request_stop()
        coord.join(threads)