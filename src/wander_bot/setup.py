from setuptools import find_packages, setup

package_name = 'wander_bot'

setup(
            name=package_name,
                version='0.0.0',
                    packages=find_packages(exclude=['test']),
                        data_files=[
                                    ('share/ament_index/resource_index/packages',
                                                 ['resource/' + package_name]),
                                            ('share/' + package_name, ['package.xml']),
                                                ],
                            install_requires=['setuptools'],
                                zip_safe=True,
                                    maintainer='user',
                                        maintainer_email='user@todo.todo',
                                            description='A package for TurtleBot3 wandering and RNN-based prediction',
                                                license='MIT',
                                                    tests_require=['pytest'],
                                                        entry_points={
                                                                    'console_scripts': [
                                                                                    'safe_wander = wander_bot.safe_wander:main',
                                                                                                'rnn_predictor = wander_bot.inference_node:main',
                                                                                                        ]
                                                                        },
                                                        )

