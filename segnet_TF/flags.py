import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('tfrecords_dir', '../tfrecords',
                            """ Folder tfrecords are stored """)
tf.app.flags.DEFINE_integer('num_images', "100",
                            """ Num images for evaluation  """)
tf.app.flags.DEFINE_integer('num_classes', "3",
                            """ Num classes """)
tf.app.flags.DEFINE_integer('max_dim', "1000",
                            """ Max dimension when downscaling """)
tf.app.flags.DEFINE_integer('fps', "1",
                            """ Frame Rate """)
tf.app.flags.DEFINE_string('save_path_video', '../009-APR-20-2-90/009-APR-20-2-90_segmented.avi',
                            """ Path to the video generated """)
tf.app.flags.DEFINE_string('video_path', '../009-APR-20-2-90/009-APR-20-2-90.MOV',
                            """ Path to video """)
tf.app.flags.DEFINE_string('caffe_weights', '../009-APR-20-2-90/caffe_weights.pickle',
                            """ caffe_weights """)
