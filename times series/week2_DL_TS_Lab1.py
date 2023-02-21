import tensorflow as tf

#Generate a TensorFlow Dataset with 10 elements (i.e number 0 to 9)
dataset = tf.data.Dataset.range(10)

#Preview the result
for val in dataset:
    print(val.numpy())

#Windowing the data
dataset = dataset.window(size=5, shift=1)

#Print the result
for window_dataset in dataset:
    print(window_dataset)
print ('next result')
#Print the result
for window_dataset in dataset:
    print([item.numpy() for item in window_dataset])

#Generate a TensorFlow Dataset with 10 elements (i.e number 0 to 9)
dataset = tf.data.Dataset.range(10)

#Windowing the data but only take those with the specified size
dataset = dataset.window(size=5, shift=1, drop_remainder=True)
print ('next result')
#Print the result
for window_dataset in dataset:
    print([item.numpy() for item in window_dataset])

# FLATTEN THE WINDOWS
#Generate a TensorFlow Dataset with 10 elements (i.e number 0 to 9)
dataset = tf.data.Dataset.range(10)

#Windowing the data but only take those with the specified size
dataset = dataset.window(size=5, shift=1, drop_remainder=True)

#Flatten the windows by putting its elements in a single batch
dataset = dataset.flat_map(lambda window: window.batch(5))
print ('next result')
#Print the result
for window_dataset in dataset:
    print(window_dataset.numpy())

#GROUP INTO FEATURES AND LABELS
#Generate a TensorFlow Dataset with 10 elements (i.e number 0 to 9)
dataset = tf.data.Dataset.range(10)

#Windowing the data but only take those with the specified size
dataset = dataset.window(size=5, shift=1, drop_remainder=True)

#Flatten the windows by putting its elements in a single batch
dataset = dataset.flat_map(lambda window: window.batch(5))

#Create tuple with features (first four elements of the windows) and labels (last element)
dataset = dataset.map(lambda window: (window[:-1], window[-1]))

print ('next result')
#Print the result
for x, y in dataset:
    print('x=', x.numpy())
    print('y=', y.numpy())
    print()

#SCHUFFLE THE DATA
#Generate a TensorFlow Dataset with 10 elements (i.e number 0 to 9)
dataset = tf.data.Dataset.range(10)

#Windowing the data but only take those with the specified size
dataset = dataset.window(size=5, shift=1, drop_remainder=True)

#Flatten the windows by putting its elements in a single batch
dataset = dataset.flat_map(lambda window: window.batch(5))

#Create tuple with features (first four elements of the windows) and labels (last element)
dataset = dataset.map(lambda window: (window[:-1], window[-1]))

#Shuffle windows
dataset = dataset.shuffle(buffer_size=10)

print ('next result')
#Print the result
for x, y in dataset:
    print('x=', x.numpy())
    print('y=', y.numpy())
    print()

#CREATES BATCH FOR TRAINING
#Generate a TensorFlow Dataset with 10 elements (i.e number 0 to 9)
dataset = tf.data.Dataset.range(10)

#Windowing the data but only take those with the specified size
dataset = dataset.window(size=5, shift=1, drop_remainder=True)

#Flatten the windows by putting its elements in a single batch
dataset = dataset.flat_map(lambda window: window.batch(5))

#Create tuple with features (first four elements of the windows) and labels (last element)
dataset = dataset.map(lambda window: (window[:-1], window[-1]))

#Shuffle windows
dataset = dataset.shuffle(buffer_size=10)

#create batch for windows
dataset = dataset.batch(2).prefetch(1)

print ('next result')
#Print the result
for x, y in dataset:
    print('x=', x.numpy())
    print('y=', y.numpy())
    print()
