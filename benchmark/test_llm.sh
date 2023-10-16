source bigdl-nano-init -c

export OMP_NUM_THREADS=52

export TRANSFORMERS_OFFLINE=1

## Chatglm2-6B

for((i=1;i<=5;i++));  
do   
echo $i ;  

numactl -C 0-51 -m 0 llm-cli -t 51 -x chatglm -m "./checkpoint/bigdl_llm_chatglm_q4_0.bin" -p "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun" --no-mmap -v -n 32

done

for((i=1;i<=5;i++));  
do   
echo $i ;  

numactl -C 0-51 -m 0 llm-cli -t 52 -x chatglm -m "./checkpoint/bigdl_llm_chatglm_q4_0.bin" -p "In the year 2048, the world was a very different place from what it had been just two decades before. The pace of technological progress had quickened to an almost unimaginable degree, and the changes that had swept through society as a result were nothing short of revolutionary. In many ways, the year 2048 represented the culmination of a long and tumultuous journey that humanity had been on since the dawn of civilization. The great leaps forward in science and technology that had occurred over the course of the previous century had laid the groundwork for a future that was beyond anything anyone could have imagined.One of the most striking aspects of life in 2048 was the degree to which technology had become an integral part of nearly every aspect of daily existence. From the moment people woke up in the morning until they went to bed at night, they were surrounded by devices and systems that were powered by advanced artificial intelligence and machine learning algorithms.In fact, it was hard to find anything in peoples lives that wasnt touched by technology in some way. Every aspect of society had been transformed, from the way people communicated with one another to the way they worked, played, and even socialized. And as the years went on, it seemed as though there was no limit to what technology could achieve.Despite all of these advances, however, not everyone was happy with the state of the world in 2048. Some people saw the increasing reliance on technology as a sign that humanity was losing touch with its own humanity, and they worried about the implications of this for the future.Others were more pragmatic, recognizing that while technology had brought many benefits, it also posed new challenges and risks that needed to be addressed. As a result, there was a growing movement of people who were working to ensure that the advances of technology were used in ways that were safe, ethical, and beneficial for everyone. One person who was at the forefront of this movement was a young woman named Maya. Maya was a brilliant and ambitious researcher who had dedicated her life to understanding the implications of emerging technologies like artificial intelligence and biotechnology. She was deeply concerned about the potential risks and unintended consequences of these technologies, and she worked tirelessly to raise awareness about the need for responsible innovation. Mayas work had earned her a reputation as one of the most influential voices in the field of technology and ethics, and she was widely respected for her deep understanding of the issues and her ability to communicate complex ideas in ways that were accessible and engaging. She was also known for her passionate and inspiring speeches, which often left her audiences with a sense of purpose and determination to make the world a better place through their own efforts. One day, Maya received an invitation to speak at a major conference on technology and ethics, which was being held in a large convention center in the heart of the city. The conference was expected to attract thousands of people from all over the world, and there was a great deal of excitement and anticipation about what Maya would say. As she prepared for her speech, Maya knew that she had a big responsibility on her shoulders. She felt a deep sense of obligation to use her platform to inspire others to take action and make a difference in the world, and she was determined to do everything in her power to live up to this responsibility. When the day of the conference arrived, Maya was filled with a mixture of excitement and nerves. She spent hours rehearsing her speech and fine-tuning her ideas, making sure that she had everything just right. Finally, after what felt like an eternity, it was time for her to take the stage. As she stepped up to the podium, Maya could feel the energy of the crowd surging around her. She took a deep breath and began to speak, her voice strong and clear as she outlined the challenges and opportunities facing society in the age of technology. She spoke passionately about the need for responsible innovation and the importance of considering the ethical implications of our actions, and she inspired many people in the audience to take up this cause and make a difference in their own lives. Overall, Mayas speech was a resounding success, and she received countless messages of gratitude and appreciation from those who had heard her speak. She knew that there was still much work to be done, but she felt hopeful about the future and the role that technology could play in creating a better world for all.  As Maya left the stage and made her way back to her seat, she couldnt help but feel a sense of pride and accomplishment at what she had just accomplished. She knew that her words had the power to inspire others and make a real difference in the world, and she was grateful for the opportunity to have played a part in this important work.  For Maya, the future was full of promise and possibility, and she was determined to continue doing everything in her power to help create a brighter, more ethical world for everyone. As she " --no-mmap -v -n 128

done


####   LLama2-13b

for((i=1;i<=4;i++));  
do   
echo $i ;  

numactl -C 0-51 -m 0 llm-cli -t 51 -x llama -m "./checkpoint/bigdl_llm_llama_q4_0.bin" -p "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun" --no-mmap -n 32

done

for((i=1;i<=5;i++));  
do   
echo $i ;  

numactl -C 0-51 -m 0 llm-cli -t 52 -x llama -m "./checkpoint/bigdl_llm_llama_q4_0.bin" -p "In the year 2048, the world was a very different place from what it had been just two decades before. The pace of technological progress had quickened to an almost unimaginable degree, and the changes that had swept through society as a result were nothing short of revolutionary. In many ways, the year 2048 represented the culmination of a long and tumultuous journey that humanity had been on since the dawn of civilization. The great leaps forward in science and technology that had occurred over the course of the previous century had laid the groundwork for a future that was beyond anything anyone could have imagined.One of the most striking aspects of life in 2048 was the degree to which technology had become an integral part of nearly every aspect of daily existence. From the moment people woke up in the morning until they went to bed at night, they were surrounded by devices and systems that were powered by advanced artificial intelligence and machine learning algorithms. In fact, it was hard to find anything in peoples lives that wasnt touched by technology in some way. Every aspect of society had been transformed, from the way people communicated with one another to the way they worked, played, and even socialized. And as the years went on, it seemed as though there was no limit to what technology could achieve.Despite all of these advances, however, not everyone was happy with the state of the world in 2048. Some people saw the increasing reliance on technology as a sign that humanity was losing touch with its own humanity, and they worried about the implications of this for the future.Others were more pragmatic, recognizing that while technology had brought many benefits, it also posed new challenges and risks that needed to be addressed. As a result, there was a growing movement of people who were working to ensure that the advances of technology were used in ways that were safe, ethical, and beneficial for everyone." --no-mmap -n 128

done
