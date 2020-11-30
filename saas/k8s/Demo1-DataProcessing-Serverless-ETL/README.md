## DataProcessing (Serverless: OpenFaaS and ETL: Apache Nifi)

The following example ETL data pipeline extracts messages from Twitter with the NiFi Twitter processor and publishes them to Apache Kafka topic. Subsequently, a Kafka processor consumes messages in the topic, preparing and sending them to the OpenFaaS SentimentAnalysis Function, finally storing the results in an Elasticsearch index for analysis within a JupyterLab environment. This example demonstrates the ease in which Kubernetes manages all the required workloads in a distributed, highly available, monitored, and unified control plane.

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo1-DataProcessing-Serverless-ETL/pictures/8-ETL_NiFi_OpenFaaS_demo_architecture.png" width="800">

###  Sererless: OpenFaaS -> Install Sentiment Analysis

Sentiment Analysis, otherwise known as emotion recognition or opinion mining, is a form of natural language processing (NLP). NLP applies linguistics, artificial intelligence, and information engineering to natural (human) languages. This section deploys a prebuilt OpenFaaS Function container, implementing the Python library TextBlob to perform Sentiment Analysis on one or more sentences of raw text. NiFi uses the deployed Sentiment Analysis Function to analyze a real-time stream of Twitter messages tagged with keywords related to COVID-19.

Browse to the OpenFaaS UI portal (https://faas.data.davar.com) and click the DEPLOY NEW FUNCTION button in the center of the screen. Next, use the Search for Function feature and search for the term SentimentAnalysis, select the Function SentimentAnalysis, and click DEPLOY on the bottom left of the dialog.

Test serverless function:
```
# Invoke the Sentiment Analysis Function using faas-cli utility
$ echo "Kubernetes is easy" | faas-cli invoke sentimentanalysis -g https://faas.data.davar.com/ --tls-no-verify
{"polarity": 0.43333333333333335, "sentence_count": 1, "subjectivity": 0.8333333333333334}

# Finally, test public access to the new Function with cURL:
$ curl -k -X POST -d "People are kind"  https://faas.data.davar.com/function/sentimentanalysis
{"polarity": 0.6, "sentence_count": 1, "subjectivity": 0.9}
```
The OpenFaaS Sentiment Analysis Function is an excellent example of a focused, self-contained bit of processing logic deployed and managed by OpenFaaS atop Kubernetes. The OpenFaaS documentation contains a well-written set of tutorials on building, testing, and implementing Functions. Functions are a great way to extend the data platform developed in this repo continuously. 

### ETL: Apache NiFi (Example ETL Data Pipeline)

#### Upload template [nifi-demo](https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/Demo1-DataProcessing-Serverless-ETL/nifi-demo) and polulate NiFi GetTwitter processor with credentials.


<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo1-DataProcessing-Serverless-ETL/pictures/Nifi-GetTwitter-populate-credentials.png" width="800">


#### Prepare Elasticsearch

```
$ cd ./Demo1-DataProcessing-Serverless-ETL/elasticsearch
# Open another terminal and execute
$ kubectl port-forward elasticsearch-0 9200:9200 -n data
$ ./PostSentimentTemplate.sh
```
#### Dataflow

Check Kafka and Elasticsearch:

```
$ curl http://localhost:9200/sentiment-*/_search
$ kubectl exec -it kafka-client-util -n data bash
root@kafka-client-util:/# kafka-topics --zookeeper zookeeper-headless:2181 --list
root@kafka-client-util:/# kafka-console-consumer --bootstrap-server kafka:9092 --topic twitter --from-beginning -max-messages 3
```
Example ouput:
```
$ curl http://localhost:9200/sentiment-*/_search
{"took":1005,"timed_out":false,"_shards":{"total":1,"successful":1,"skipped":0,"failed":0},"hits":{"total":{"value":1208,"relation":"eq"},"max_score":1.0,"hits":[{"_index":"sentiment-2020-11","_type":"_doc","_id":"TITcF3YBZceEtiIk2_KS","_score":1.0,"_source":{"twitter.msg":"RT @MSNBC: WATCH: Experts discuss Covid-19 fatigue, the coming vaccine and the work ahead for President-elect Biden.\nhttps://t.co/en0ZZyTEC2","invokehttp.tx.id":"72953f9d-802f-45a4-8e3d-639a683f50db","X-Duration-Seconds":"0.518289","subjectivity":"0.0","kafka.partition":"0","sentence_count":"2","invokehttp.status.code":"200","mime.type":"text/csv","uuid":"e08f7c42-3bbd-4d79-8724-28a829eba753","Date":"Mon, 30 Nov 2020 06:35:17 GMT","twitter.handle":"vmrwanda","invokehttp.request.url":"http://gateway:8080/function/sentimentanalysis","path":"./","filename":"1d70fca3-631d-4d52-a388-99bdd71d2cbc","twitter.user":"Victor Rwanda","kafka.offset":"0","kafka.topic":"twitter","twitter.created_at":"Mon Nov 30 06:35:09 +0000 2020","invokehttp.status.message":"OK","Content-Length":"60","RouteOnAttribute.Route":"matched","Content-Type":"text/csv","polarity":"0.0"}},{"_index":"sentiment-2020-11","_type":"_doc","_id":"TYTcF3YBZceEtiIk3PIX","_score":1.0,"_source":{"twitter.msg":"RT @WIRED: At first, it appeared venture capitalists responded swiftly to meet the Covid-19 challenge, investing heavily in education techn…","invokehttp.tx.id":"74811258-f11b-45be-98f8-9a8fcca4d8d7","X-Duration-Seconds":"0.344682","subjectivity":"0.41666666666666663","kafka.partition":"0","sentence_count":"1","invokehttp.status.code":"200","mime.type":"text/csv","uuid":"5fd381a4-c27b-47b2-8433-72f45ff7cc06","Date":"Mon, 30 Nov 2020 06:35:18 GMT","twitter.handle":"taiesalami","invokehttp.request.url":"http://gateway:8080/function/sentimentanalysis","path":"./","filename":"08a3fe7c-5bda-42c9-9ff4-2e4a37ece211","twitter.user":"Taiye Salami","kafka.offset":"1","kafka.topic":"twitter","twitter.created_at":"Mon Nov 30 06:35:09 +0000 2020","invokehttp.status.message":"OK","Content-Length":"93","RouteOnAttribute.Route":"matched","Content-Type":"text/csv","polarity":"0.024999999999999994"}},{"_index":"sentiment-2020-11","_type":"_doc","_id":"ToTcF3YBZceEtiIk3PIX","_score":1.0,"_source":{"twitter.msg":"RT @KEPSA_KENYA: The MSME Covid19 Recovery and Resilience program loans are now accessible through our online portal https://t.co/aw3NcbCs3…","invokehttp.tx.id":"d977034c-c51c-42e2-af4c-55d204b664ed","X-Duration-Seconds":"0.410955","subjectivity":"0.375","kafka.partition":"0","sentence_count":"1","invokehttp.status.code":"200","mime.type":"text/csv","uuid":"87666efc-4152-4a51-9d23-60b531305f4b","Date":"Mon, 30 Nov 2020 06:35:18 GMT","twitter.handle":"Kuria_Kungu1","invokehttp.request.url":"http://gateway:8080/function/sentimentanalysis","path":"./","filename":"e2c72018-511e-4ebc-aee0-f78c3c65dfbe","twitter.user":"James","kafka.offset":"2","kafka.topic":"twitter","twitter.created_at":"Mon Nov 30 06:35:09 +0000 2020","invokehttp.status.message":"OK","Content-Length":"64","RouteOnAttribute.Route":"matched","Content-Type":"text/csv","polarity":"0.375"}},{"_index":"sentiment-2020-11","_type":"_doc","_id":"YITdF3YBZceEtiIkH_I_","_score":1.0,"_source":{"twitter.msg":"RT @RileyBehrens: Earlier today, I was diagnosed as having suffered a Transient Ischemic Attack (TIA), or what's commonly known as a mini-s…","invokehttp.tx.id":"397b3d04-88af-4106-9a8d-7c60273ef6aa","X-Duration-Seconds":"0.286946","subjectivity":"0.5","kafka.partition":"0","sentence_count":"1","invokehttp.status.code":"200","mime.type":"text/csv","uuid":"16288029-9485-4e67-9d03-8ea8263c4fe1","Date":"Mon, 30 Nov 2020 06:35:36 GMT","twitter.handle":"_VuyoHlwatika","invokehttp.request.url":"http://gateway:8080/function/sentimentanalysis","path":"./","filename":"085678cd-a3a8-43bf-93e2-69e618577452","twitter.user":"The NJE is silent\uD83C\uDFF3️‍\uD83C\uDF08","kafka.offset":"20","kafka.topic":"twitter","twitter.created_at":"Mon Nov 30 06:35:12 +0000 2020","invokehttp.status.message":"OK","Content-Length":"62","RouteOnAttribute.Route":"matched","Content-Type":"text/csv","polarity":"-0.15"}},{"_index":"sentiment-2020-11","_type":"_doc","_id":"X4TdF3YBZceEtiIkG_I3","_score":1.0,"_source":{"twitter.msg":"RT @theseoulstory: AKMU, TREASURE and their staff members to undergo COVID-19 testing today\n\nThey performed on MBC ‘Music Core’ and SBS 'In…","invokehttp.tx.id":"bf103df3-7dc5-4bc4-b25e-89e5785e73aa","X-Duration-Seconds":"0.288369","subjectivity":"0.0","kafka.partition":"0","sentence_count":"1","invokehttp.status.code":"200","mime.type":"text/csv","uuid":"f88368bc-73e8-4210-9758-112b22e5bc0c","Date":"Mon, 30 Nov 2020 06:35:35 GMT","twitter.handle":"tylightbin","invokehttp.request.url":"http://gateway:8080/function/sentimentanalysis","path":"./","filename":"c8d5c872-a6ef-4d7c-84e7-7e044f2197a4","twitter.user":"тιαяα\uD83C\uDF39\uD83C\uDF38","kafka.offset":"19","kafka.topic":"twitter","twitter.created_at":"Mon Nov 30 06:35:11 +0000 2020","invokehttp.status.message":"OK","Content-Length":"60","RouteOnAttribute.Route":"matched","Content-Type":"text/csv","polarity":"0.0"}},{"_index":"sentiment-2020-11","_type":"_doc","_id":"YYTdF3YBZceEtiIkI_Jj","_score":1.0,"_source":{"twitter.msg":"RT @RileyBehrens: Earlier today, I was diagnosed as having suffered a Transient Ischemic Attack (TIA), or what's commonly known as a mini-s…","invokehttp.tx.id":"514d642c-f9ec-45b7-9ad5-23d3ceb97c85","X-Duration-Seconds":"0.309646","subjectivity":"0.5","kafka.partition":"0","sentence_count":"1","invokehttp.status.code":"200","mime.type":"text/csv","uuid":"066af27c-fe5a-452a-9bde-ae8bd3cea176","Date":"Mon, 30 Nov 2020 06:35:37 GMT","twitter.handle":"iihatebums21","invokehttp.request.url":"http://gateway:8080/function/sentimentanalysis","path":"./","filename":"01302239-ed7a-41f6-a38e-51b607b33e74","twitter.user":"❤Shirley Ryan❤","kafka.offset":"21","kafka.topic":"twitter","twitter.created_at":"Mon Nov 30 06:35:12 +0000 2020","invokehttp.status.message":"OK","Content-Length":"62","RouteOnAttribute.Route":"matched","Content-Type":"text/csv","polarity":"-0.15"}},{"_index":"sentiment-2020-11","_type":"_doc","_id":"XoTdF3YBZceEtiIkF_JN","_score":1.0,"_source":{"twitter.msg":"RT @Craig_A_Spencer: As COVID19 surges across the US, it’s hard to describe the situation inside hospitals for healthcare providers &amp; patie…","invokehttp.tx.id":"6f9f6e38-9e17-42f8-b749-e8be92d45d34","X-Duration-Seconds":"0.295084","subjectivity":"0.5416666666666666","kafka.partition":"0","sentence_count":"1","invokehttp.status.code":"200","mime.type":"text/csv","uuid":"087f0a97-b09c-4ccc-a0f5-c2366ce02501","Date":"Mon, 30 Nov 2020 06:35:34 GMT","twitter.handle":"TGJR777","invokehttp.request.url":"http://gateway:8080/function/sentimentanalysis","path":"./","filename":"900ad050-cba0-47e5-92fc-bf2df1114986","twitter.user":"Pizzaball Courier","kafka.offset":"18","kafka.topic":"twitter","twitter.created_at":"Mon Nov 30 06:35:11 +0000 2020","invokehttp.status.message":"OK","Content-Length":"91","RouteOnAttribute.Route":"matched","Content-Type":"text/csv","polarity":"-0.2916666666666667"}},{"_index":"sentiment-2020-11","_type":"_doc","_id":"XITdF3YBZceEtiIkD_JF","_score":1.0,"_source":{"twitter.msg":"RT @breastitude: COVID-19 Update For November 29 2020 In Nigeria https://t.co/sdhI21h2Vu\n#EndNaijaKillings #ZabarmariMassacre #MondayMotiva…","invokehttp.tx.id":"191af22d-5aef-45b0-94ec-17cd5c57560b","X-Duration-Seconds":"0.299284","subjectivity":"0.0","kafka.partition":"0","sentence_count":"1","invokehttp.status.code":"200","mime.type":"text/csv","uuid":"b348f176-6716-46e6-a7cc-bdfb2e1014e5","Date":"Mon, 30 Nov 2020 06:35:32 GMT","twitter.handle":"FlipmemesDotCom","invokehttp.request.url":"http://gateway:8080/function/sentimentanalysis","path":"./","filename":"092376f0-bc61-4093-bcb5-a05b21ca9994","twitter.user":"Flipmemes","kafka.offset":"16","kafka.topic":"twitter","twitter.created_at":"Mon Nov 30 06:35:11 +0000 2020","invokehttp.status.message":"OK","Content-Length":"60","RouteOnAttribute.Route":"matched","Content-Type":"text/csv","polarity":"0.0"}},{"_index":"sentiment-2020-11","_type":"_doc","_id":"WoTdF3YBZceEtiIkB_JP","_score":1.0,"_source":{"twitter.msg":"RT @balloon_wanted: All of SBS Inkigayo MCs including MONSTA X's Minhyuk, NCT's Jaehyun, and APRIL's Naeun will be undergoing COVID-19 test…","invokehttp.tx.id":"1832118c-98d9-4ed8-b15e-6ab1fe250a9a","X-Duration-Seconds":"0.316942","subjectivity":"0.0","kafka.partition":"0","sentence_count":"1","invokehttp.status.code":"200","mime.type":"text/csv","uuid":"a6872d5f-aceb-4b37-b63d-6e01f298bffa","Date":"Mon, 30 Nov 2020 06:35:29 GMT","twitter.handle":"schgmk","invokehttp.request.url":"http://gateway:8080/function/sentimentanalysis","path":"./","filename":"1cb5cbb0-407b-4239-b9e2-1c996d19c1be","twitter.user":"࣪ #RESPECTWAYV","kafka.offset":"14","kafka.topic":"twitter","twitter.created_at":"Mon Nov 30 06:35:11 +0000 2020","invokehttp.status.message":"OK","Content-Length":"60","RouteOnAttribute.Route":"matched","Content-Type":"text/csv","polarity":"0.0"}},{"_index":"sentiment-2020-11","_type":"_doc","_id":"XYTdF3YBZceEtiIkE_I5","_score":1.0,"_source":{"twitter.msg":"“If the cutouts could come to life...” \uD83D\uDE0A\uD83C\uDFC8 #BeaverNation #beaverfootball #OregonStateUniversity #COVID19","invokehttp.tx.id":"22de0f58-0e25-43d8-93d3-490083fec109","X-Duration-Seconds":"0.291195","subjectivity":"0.0","kafka.partition":"0","sentence_count":"1","invokehttp.status.code":"200","mime.type":"text/csv","uuid":"e7a4f82a-0c8f-4b60-ab12-ed3ce9aa6519","Date":"Mon, 30 Nov 2020 06:35:33 GMT","twitter.handle":"GTaylor31447443","invokehttp.request.url":"http://gateway:8080/function/sentimentanalysis","path":"./","filename":"e9cc44ba-62d6-454d-924b-5f06de459a0d","twitter.user":"GTaylor","kafka.offset":"17","kafka.topic":"twitter","twitter.created_at":"Mon Nov 30 06:35:11 +0000 2020","invokehttp.status.message":"OK","Content-Length":"60","RouteOnAttribute.Route":"matched","Content-Type":"text/csv","polarity":"0.0"}}]}}

$ kubectl exec -it kafka-client-util -n data bash
root@kafka-client-util:/# kafka-topics --zookeeper zookeeper-headless:2181 --list
__confluent.support.metrics
__consumer_offsets
messages
metrics
twitter

root@kafka-client-util:/# kafka-console-consumer --bootstrap-server kafka:9092 --topic twitter --from-beginning -max-messages 3
{"created_at":"Mon Nov 30 06:35:09 +0000 2020","id":1333298504178012160,"id_str":"1333298504178012160","text":"RT @MSNBC: WATCH: Experts discuss Covid-19 fatigue, the coming vaccine and the work ahead for President-elect Biden.\nhttps:\/\/t.co\/en0ZZyTEC2","source":"\u003ca href=\"https:\/\/pesarika.co.ke\" rel=\"nofollow\"\u003eCorona Updates EA\u003c\/a\u003e","truncated":false,"in_reply_to_status_id":null,"in_reply_to_status_id_str":null,"in_reply_to_user_id":null,"in_reply_to_user_id_str":null,"in_reply_to_screen_name":null,"user":{"id":292898515,"id_str":"292898515","name":"Victor Rwanda","screen_name":"vmrwanda","location":null,"url":null,"description":"Software Engineer","translator_type":"none","protected":false,"verified":false,"followers_count":906,"friends_count":908,"listed_count":20,"favourites_count":8511,"statuses_count":188441,"created_at":"Wed May 04 12:31:42 +0000 2011","utc_offset":null,"time_zone":null,"geo_enabled":true,"lang":null,"contributors_enabled":false,"is_translator":false,"profile_background_color":"000000","profile_background_image_url":"http:\/\/abs.twimg.com\/images\/themes\/theme1\/bg.png","profile_background_image_url_https":"https:\/\/abs.twimg.com\/images\/themes\/theme1\/bg.png","profile_background_tile":false,"profile_link_color":"E81C4F","profile_sidebar_border_color":"000000","profile_sidebar_fill_color":"000000","profile_text_color":"000000","profile_use_background_image":false,"profile_image_url":"http:\/\/pbs.twimg.com\/profile_images\/1317837830698291200\/cCyTVZQI_normal.jpg","profile_image_url_https":"https:\/\/pbs.twimg.com\/profile_images\/1317837830698291200\/cCyTVZQI_normal.jpg","profile_banner_url":"https:\/\/pbs.twimg.com\/profile_banners\/292898515\/1603031914","default_profile":false,"default_profile_image":false,"following":null,"follow_request_sent":null,"notifications":null},"geo":null,"coordinates":null,"place":null,"contributors":null,"retweeted_status":{"created_at":"Mon Nov 30 06:33:02 +0000 2020","id":1333297970192715777,"id_str":"1333297970192715777","text":"WATCH: Experts discuss Covid-19 fatigue, the coming vaccine and the work ahead for President-elect Biden.\nhttps:\/\/t.co\/en0ZZyTEC2","source":"\u003ca href=\"http:\/\/www.socialflow.com\" rel=\"nofollow\"\u003eSocialFlow\u003c\/a\u003e","truncated":false,"in_reply_to_status_id":null,"in_reply_to_status_id_str":null,"in_reply_to_user_id":null,"in_reply_to_user_id_str":null,"in_reply_to_screen_name":null,"user":{"id":2836421,"id_str":"2836421","name":"MSNBC","screen_name":"MSNBC","location":null,"url":"http:\/\/msnbc.com\/live","description":"#MSNBC2020: The place for in-depth analysis, political commentary and diverse perspectives. Home of @MSNBCDaily.","translator_type":"regular","protected":false,"verified":true,"followers_count":3884486,"friends_count":737,"listed_count":26765,"favourites_count":819,"statuses_count":228766,"created_at":"Thu Mar 29 13:15:41 +0000 2007","utc_offset":null,"time_zone":null,"geo_enabled":true,"lang":null,"contributors_enabled":false,"is_translator":false,"profile_background_color":"000000","profile_background_image_url":"http:\/\/abs.twimg.com\/images\/themes\/theme1\/bg.png","profile_background_image_url_https":"https:\/\/abs.twimg.com\/images\/themes\/theme1\/bg.png","profile_background_tile":false,"profile_link_color":"0CB1C7","profile_sidebar_border_color":"FFFFFF","profile_sidebar_fill_color":"EEEEEE","profile_text_color":"000000","profile_use_background_image":true,"profile_image_url":"http:\/\/pbs.twimg.com\/profile_images\/1321089811290202119\/6QEBEjWk_normal.jpg","profile_image_url_https":"https:\/\/pbs.twimg.com\/profile_images\/1321089811290202119\/6QEBEjWk_normal.jpg","profile_banner_url":"https:\/\/pbs.twimg.com\/profile_banners\/2836421\/1585585008","default_profile":false,"default_profile_image":false,"following":null,"follow_request_sent":null,"notifications":null},"geo":null,"coordinates":null,"place":null,"contributors":null,"is_quote_status":false,"quote_count":2,"reply_count":4,"retweet_count":5,"favorite_count":11,"entities":{"hashtags":[],"urls":[{"url":"https:\/\/t.co\/en0ZZyTEC2","expanded_url":"https:\/\/on.msnbc.com\/2JqYFTg","display_url":"on.msnbc.com\/2JqYFTg","indices":[106,129]}],"user_mentions":[],"symbols":[]},"favorited":false,"retweeted":false,"possibly_sensitive":false,"filter_level":"low","lang":"en"},"is_quote_status":false,"quote_count":0,"reply_count":0,"retweet_count":0,"favorite_count":0,"entities":{"hashtags":[],"urls":[{"url":"https:\/\/t.co\/en0ZZyTEC2","expanded_url":"https:\/\/on.msnbc.com\/2JqYFTg","display_url":"on.msnbc.com\/2JqYFTg","indices":[117,140]}],"user_mentions":[{"screen_name":"MSNBC","name":"MSNBC","id":2836421,"id_str":"2836421","indices":[3,9]}],"symbols":[]},"favorited":false,"retweeted":false,"possibly_sensitive":false,"filter_level":"low","lang":"en","timestamp_ms":"1606718109541"}

{"created_at":"Mon Nov 30 06:35:09 +0000 2020","id":1333298505159483393,"id_str":"1333298505159483393","text":"RT @WIRED: At first, it appeared venture capitalists responded swiftly to meet the Covid-19 challenge, investing heavily in education techn\u2026","source":"\u003ca href=\"http:\/\/twitter.com\/download\/android\" rel=\"nofollow\"\u003eTwitter for Android\u003c\/a\u003e","truncated":false,"in_reply_to_status_id":null,"in_reply_to_status_id_str":null,"in_reply_to_user_id":null,"in_reply_to_user_id_str":null,"in_reply_to_screen_name":null,"user":{"id":2340898859,"id_str":"2340898859","name":"Taiye Salami","screen_name":"taiesalami","location":"Nigeria | Worldwide","url":null,"description":"Programs, @AfriLabs | Innovation, @GlobalEdTechHub\n\nEgg-head | Twin | I may or may not be joking but...Aal Izz Well |","translator_type":"none","protected":false,"verified":false,"followers_count":438,"friends_count":3940,"listed_count":1,"favourites_count":1230,"statuses_count":2077,"created_at":"Thu Feb 13 08:49:12 +0000 2014","utc_offset":null,"time_zone":null,"geo_enabled":true,"lang":null,"contributors_enabled":false,"is_translator":false,"profile_background_color":"000000","profile_background_image_url":"http:\/\/abs.twimg.com\/images\/themes\/theme1\/bg.png","profile_background_image_url_https":"https:\/\/abs.twimg.com\/images\/themes\/theme1\/bg.png","profile_background_tile":false,"profile_link_color":"ABB8C2","profile_sidebar_border_color":"000000","profile_sidebar_fill_color":"000000","profile_text_color":"000000","profile_use_background_image":false,"profile_image_url":"http:\/\/pbs.twimg.com\/profile_images\/1315199960787415049\/5NeikX_F_normal.jpg","profile_image_url_https":"https:\/\/pbs.twimg.com\/profile_images\/1315199960787415049\/5NeikX_F_normal.jpg","profile_banner_url":"https:\/\/pbs.twimg.com\/profile_banners\/2340898859\/1429556995","default_profile":false,"default_profile_image":false,"following":null,"follow_request_sent":null,"notifications":null},"geo":null,"coordinates":null,"place":null,"contributors":null,"retweeted_status":{"created_at":"Mon Nov 30 03:06:26 +0000 2020","id":1333245977441021954,"id_str":"1333245977441021954","text":"At first, it appeared venture capitalists responded swiftly to meet the Covid-19 challenge, investing heavily in ed\u2026 https:\/\/t.co\/sX0UfE8jg6","source":"\u003ca href=\"http:\/\/www.socialflow.com\" rel=\"nofollow\"\u003eSocialFlow\u003c\/a\u003e","truncated":true,"in_reply_to_status_id":null,"in_reply_to_status_id_str":null,"in_reply_to_user_id":null,"in_reply_to_user_id_str":null,"in_reply_to_screen_name":null,"user":{"id":1344951,"id_str":"1344951","name":"WIRED","screen_name":"WIRED","location":"San Francisco\/New York","url":"http:\/\/www.wired.com","description":"WIRED is where tomorrow is realized.","translator_type":"none","protected":false,"verified":true,"followers_count":10384624,"friends_count":401,"listed_count":89362,"favourites_count":3671,"statuses_count":121663,"created_at":"Sat Mar 17 09:57:25 +0000 2007","utc_offset":null,"time_zone":null,"geo_enabled":true,"lang":null,"contributors_enabled":false,"is_translator":false,"profile_background_color":"000000","profile_background_image_url":"http:\/\/abs.twimg.com\/images\/themes\/theme1\/bg.png","profile_background_image_url_https":"https:\/\/abs.twimg.com\/images\/themes\/theme1\/bg.png","profile_background_tile":false,"profile_link_color":"99DCF0","profile_sidebar_border_color":"FFFFFF","profile_sidebar_fill_color":"EEEEEE","profile_text_color":"000000","profile_use_background_image":false,"profile_image_url":"http:\/\/pbs.twimg.com\/profile_images\/1228050699348561920\/YvWAQD2L_normal.jpg","profile_image_url_https":"https:\/\/pbs.twimg.com\/profile_images\/1228050699348561920\/YvWAQD2L_normal.jpg","profile_banner_url":"https:\/\/pbs.twimg.com\/profile_banners\/1344951\/1605021048","default_profile":false,"default_profile_image":false,"following":null,"follow_request_sent":null,"notifications":null},"geo":null,"coordinates":null,"place":null,"contributors":null,"is_quote_status":false,"extended_tweet":{"full_text":"At first, it appeared venture capitalists responded swiftly to meet the Covid-19 challenge, investing heavily in education technology. But a closer look reveals cause for concern, particularly with respect to funding for educationally vulnerable students. https:\/\/t.co\/POgxeIrzkh","display_text_range":[0,279],"entities":{"hashtags":[],"urls":[{"url":"https:\/\/t.co\/POgxeIrzkh","expanded_url":"https:\/\/wired.trib.al\/9sGxeWX","display_url":"wired.trib.al\/9sGxeWX","indices":[256,279]}],"user_mentions":[],"symbols":[]}},"quote_count":3,"reply_count":4,"retweet_count":17,"favorite_count":55,"entities":{"hashtags":[],"urls":[{"url":"https:\/\/t.co\/sX0UfE8jg6","expanded_url":"https:\/\/twitter.com\/i\/web\/status\/1333245977441021954","display_url":"twitter.com\/i\/web\/status\/1\u2026","indices":[117,140]}],"user_mentions":[],"symbols":[]},"favorited":false,"retweeted":false,"possibly_sensitive":false,"filter_level":"low","lang":"en"},"is_quote_status":false,"quote_count":0,"reply_count":0,"retweet_count":0,"favorite_count":0,"entities":{"hashtags":[],"urls":[],"user_mentions":[{"screen_name":"WIRED","name":"WIRED","id":1344951,"id_str":"1344951","indices":[3,9]}],"symbols":[]},"favorited":false,"retweeted":false,"filter_level":"low","lang":"en","timestamp_ms":"1606718109775"}

{"created_at":"Mon Nov 30 06:35:09 +0000 2020","id":1333298505960583168,"id_str":"1333298505960583168","text":"RT @KEPSA_KENYA: The MSME Covid19 Recovery and Resilience program loans are now accessible through our online portal https:\/\/t.co\/aw3NcbCs3\u2026","source":"\u003ca href=\"http:\/\/twitter.com\/download\/android\" rel=\"nofollow\"\u003eTwitter for Android\u003c\/a\u003e","truncated":false,"in_reply_to_status_id":null,"in_reply_to_status_id_str":null,"in_reply_to_user_id":null,"in_reply_to_user_id_str":null,"in_reply_to_screen_name":null,"user":{"id":636463866,"id_str":"636463866","name":"James","screen_name":"Kuria_Kungu1","location":"Nakuru, Kenya","url":"http:\/\/www.kuriakungu.wordpress.com","description":"Dextrous","translator_type":"none","protected":false,"verified":false,"followers_count":48,"friends_count":128,"listed_count":1,"favourites_count":207,"statuses_count":185,"created_at":"Sun Jul 15 19:03:43 +0000 2012","utc_offset":null,"time_zone":null,"geo_enabled":true,"lang":null,"contributors_enabled":false,"is_translator":false,"profile_background_color":"0099B9","profile_background_image_url":"http:\/\/abs.twimg.com\/images\/themes\/theme1\/bg.png","profile_background_image_url_https":"https:\/\/abs.twimg.com\/images\/themes\/theme1\/bg.png","profile_background_tile":true,"profile_link_color":"19CF86","profile_sidebar_border_color":"FFFFFF","profile_sidebar_fill_color":"DDEEF6","profile_text_color":"333333","profile_use_background_image":true,"profile_image_url":"http:\/\/pbs.twimg.com\/profile_images\/784450842707525632\/JPtlP6_j_normal.jpg","profile_image_url_https":"https:\/\/pbs.twimg.com\/profile_images\/784450842707525632\/JPtlP6_j_normal.jpg","profile_banner_url":"https:\/\/pbs.twimg.com\/profile_banners\/636463866\/1475862838","default_profile":false,"default_profile_image":false,"following":null,"follow_request_sent":null,"notifications":null},"geo":null,"coordinates":null,"place":null,"contributors":null,"retweeted_status":{"created_at":"Mon Nov 30 05:27:50 +0000 2020","id":1333281562985631744,"id_str":"1333281562985631744","text":"The MSME Covid19 Recovery and Resilience program loans are now accessible through our online portal\u2026 https:\/\/t.co\/tBUNGWZtTI","display_text_range":[0,140],"source":"\u003ca href=\"https:\/\/mobile.twitter.com\" rel=\"nofollow\"\u003eTwitter Web App\u003c\/a\u003e","truncated":true,"in_reply_to_status_id":null,"in_reply_to_status_id_str":null,"in_reply_to_user_id":null,"in_reply_to_user_id_str":null,"in_reply_to_screen_name":null,"user":{"id":497878671,"id_str":"497878671","name":"KEPSA KENYA","screen_name":"KEPSA_KENYA","location":"Nairobi, Kenya","url":"http:\/\/www.kepsa.or.ke","description":"KEPSA is the national apex body of the private sector in Kenya comprising of Business Associations and Corporate Organizations.","translator_type":"none","protected":false,"verified":true,"followers_count":40761,"friends_count":2173,"listed_count":72,"favourites_count":1524,"statuses_count":9562,"created_at":"Mon Feb 20 13:13:43 +0000 2012","utc_offset":null,"time_zone":null,"geo_enabled":true,"lang":null,"contributors_enabled":false,"is_translator":false,"profile_background_color":"C0DEED","profile_background_image_url":"http:\/\/abs.twimg.com\/images\/themes\/theme1\/bg.png","profile_background_image_url_https":"https:\/\/abs.twimg.com\/images\/themes\/theme1\/bg.png","profile_background_tile":false,"profile_link_color":"1DA1F2","profile_sidebar_border_color":"C0DEED","profile_sidebar_fill_color":"DDEEF6","profile_text_color":"333333","profile_use_background_image":true,"profile_image_url":"http:\/\/pbs.twimg.com\/profile_images\/507512965608636416\/KCkktqRp_normal.png","profile_image_url_https":"https:\/\/pbs.twimg.com\/profile_images\/507512965608636416\/KCkktqRp_normal.png","profile_banner_url":"https:\/\/pbs.twimg.com\/profile_banners\/497878671\/1603788274","default_profile":true,"default_profile_image":false,"following":null,"follow_request_sent":null,"notifications":null},"geo":null,"coordinates":null,"place":null,"contributors":null,"is_quote_status":false,"extended_tweet":{"full_text":"The MSME Covid19 Recovery and Resilience program loans are now accessible through our online portal https:\/\/t.co\/aw3NcbCs3z where you are required to create an account, register your business (MSME) then apply for the loan. For Enquiries call: +254 773 132 403\n#KEPSAnaBiashara https:\/\/t.co\/FDPq6pbPkS","display_text_range":[0,277],"entities":{"hashtags":[{"text":"KEPSAnaBiashara","indices":[261,277]}],"urls":[{"url":"https:\/\/t.co\/aw3NcbCs3z","expanded_url":"http:\/\/smekepsa.or.ke\/","display_url":"smekepsa.or.ke","indices":[100,123]}],"user_mentions":[],"symbols":[],"media":[{"id":1333281513912340480,"id_str":"1333281513912340480","indices":[278,301],"media_url":"http:\/\/pbs.twimg.com\/media\/EoDEeQ5XEAA-sJF.jpg","media_url_https":"https:\/\/pbs.twimg.com\/media\/EoDEeQ5XEAA-sJF.jpg","url":"https:\/\/t.co\/FDPq6pbPkS","display_url":"pic.twitter.com\/FDPq6pbPkS","expanded_url":"https:\/\/twitter.com\/KEPSA_KENYA\/status\/1333281562985631744\/photo\/1","type":"photo","sizes":{"thumb":{"w":150,"h":150,"resize":"crop"},"small":{"w":680,"h":299,"resize":"fit"},"medium":{"w":1200,"h":527,"resize":"fit"},"large":{"w":1205,"h":529,"resize":"fit"}}}]},"extended_entities":{"media":[{"id":1333281513912340480,"id_str":"1333281513912340480","indices":[278,301],"media_url":"http:\/\/pbs.twimg.com\/media\/EoDEeQ5XEAA-sJF.jpg","media_url_https":"https:\/\/pbs.twimg.com\/media\/EoDEeQ5XEAA-sJF.jpg","url":"https:\/\/t.co\/FDPq6pbPkS","display_url":"pic.twitter.com\/FDPq6pbPkS","expanded_url":"https:\/\/twitter.com\/KEPSA_KENYA\/status\/1333281562985631744\/photo\/1","type":"photo","sizes":{"thumb":{"w":150,"h":150,"resize":"crop"},"small":{"w":680,"h":299,"resize":"fit"},"medium":{"w":1200,"h":527,"resize":"fit"},"large":{"w":1205,"h":529,"resize":"fit"}}}]}},"quote_count":1,"reply_count":0,"retweet_count":8,"favorite_count":6,"entities":{"hashtags":[],"urls":[{"url":"https:\/\/t.co\/tBUNGWZtTI","expanded_url":"https:\/\/twitter.com\/i\/web\/status\/1333281562985631744","display_url":"twitter.com\/i\/web\/status\/1\u2026","indices":[101,124]}],"user_mentions":[],"symbols":[]},"favorited":false,"retweeted":false,"possibly_sensitive":false,"filter_level":"low","lang":"en"},"is_quote_status":false,"quote_count":0,"reply_count":0,"retweet_count":0,"favorite_count":0,"entities":{"hashtags":[],"urls":[],"user_mentions":[{"screen_name":"KEPSA_KENYA","name":"KEPSA KENYA","id":497878671,"id_str":"497878671","indices":[3,15]}],"symbols":[]},"favorited":false,"retweeted":false,"filter_level":"low","lang":"en","timestamp_ms":"1606718109966"}

Processed a total of 3 messages
```

Elasticsearch Aggreagate query:


```
$ ./PostSentimentAnalysisQuery.sh
```

The following query aggregates the last hour of the polarity metric from Sentiment Analysis into histogram buckets at every 0.5 interval from -1 to 1. Elasticsearch supports a robust set of aggregation capabilities.

Example output:
```
./PostSentimentAnalysisQuery.sh |  python -m json.tool
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   659  100   336  100   323    962    925 --:--:-- --:--:-- --:--:--  1888
{
  "took": 5,
  "timed_out": false,
  "_shards": {
    "total": 1,
    "successful": 1,
    "skipped": 0,
    "failed": 0
  },
  "hits": {
    "total": {
      "value": 2276,
      "relation": "eq"
    },
    "max_score": null,
    "hits": []
  },
  "aggregations": {
    "polarity": {
      "buckets": [
        {
          "key": -1,
          "doc_count": 40
        },
        
        {
          "key": -0.5,
          "doc_count":
        },
        {
          "key": 0,
          "doc_count":
        },
        {
          "key": 0.5,
          "doc_count":
        },
        {
          "key": 1,
          "doc_count":
        }
      ]
    }
  }
}
```
The example results show there were more negative Twitter posts ("doc_count": 40) regarding COVID-19 than
positive ("doc_count": 4) in the last hour.


### JupyterLab 

Jupyter Notebooks are a browser-based (or web-based) IDE (integrated development environments)

Build custom JupyterLab docker image
```
$ cd ./JupyterLab
$ docker build -t jupyterlab-eth .
$ docker tag jupyterlab-eth:latest davarski/jupyterlab-eth:latest
$ docker login 
$ docker push davarski/jupyterlab-eth:latest
```
Run Jupyter Notebook

```
$ sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/k3s-config-jupyter
$ sed -i "s/127.0.0.1/192.168.0.101/" ~/.kube/k3s-config-jupyter
$ docker run --rm --name jl -p 8888:8888 \
   -v "$(pwd)":"/home/jovyan/work" \
   -v "$HOME/.kube/k3s-config-jupyter":"/home/jovyan/.kube/config" \
   --user root \
   -e GRANT_SUDO=yes \
   -e JUPYTER_ENABLE_LAB=yes -e RESTARTABLE=yes \
   davarski/jupyterlab-eth:latest
```
Example:
```
$ docker run --rm --name jl -p 8888:8888 \
>    -v "$(pwd)":"/home/jovyan/work" \
>    -v "$HOME/.kube/k3s-config-jupyter":"/home/jovyan/.kube/config" \
>    --user root \
>    -e GRANT_SUDO=yes \
>    -e JUPYTER_ENABLE_LAB=yes -e RESTARTABLE=yes \
>    davarski/jupyterlab-eth:latest

Set username to: jovyan
usermod: no changes
Granting jovyan sudo access and appending /opt/conda/bin to sudo PATH
Executing the command: jupyter lab
[I 21:37:15.811 LabApp] Writing notebook server cookie secret to /home/jovyan/.local/share/jupyter/runtime/notebook_cookie_secret
[I 21:37:16.594 LabApp] Loading IPython parallel extension
[I 21:37:16.614 LabApp] JupyterLab extension loaded from /opt/conda/lib/python3.7/site-packages/jupyterlab
[I 21:37:16.614 LabApp] JupyterLab application directory is /opt/conda/share/jupyter/lab
[W 21:37:16.623 LabApp] JupyterLab server extension not enabled, manually loading...
[I 21:37:16.638 LabApp] JupyterLab extension loaded from /opt/conda/lib/python3.7/site-packages/jupyterlab
[I 21:37:16.638 LabApp] JupyterLab application directory is /opt/conda/share/jupyter/lab
[I 21:37:16.639 LabApp] Serving notebooks from local directory: /home/jovyan
[I 21:37:16.639 LabApp] The Jupyter Notebook is running at:
[I 21:37:16.639 LabApp] http://(e1696ffe20ab or 127.0.0.1):8888/?token=f0c6d63a7ffb4e67d132716e3ed49745e97b3e7fa78db28d
[I 21:37:16.639 LabApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 21:37:16.648 LabApp] 
    
    To access the notebook, open this file in a browser:
        file:///home/jovyan/.local/share/jupyter/runtime/nbserver-17-open.html
    Or copy and paste one of these URLs:
        http://(e1696ffe20ab or 127.0.0.1):8888/?token=f0c6d63a7ffb4e67d132716e3ed49745e97b3e7fa78db28d
```
Open IDE in browser: http://127.0.0.1:8888/?token=f0c6d63a7ffb4e67d132716e3ed49745e97b3e7fa78db28d

Within the Docker container (at localhost:8888), under the section titled Other within the running Jupyter Notebook, chose Terminal. Once the terminal launches, provide the following command to port-forward all services running in the data Namespace on the k8s cluster 
```
sudo kubefwd svc -n data
```
<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo1-DataProcessing-Serverless-ETL/pictures/ETL-JupyterLab-forward-svc.png" width="800">

The utility kubefwd connects and port-forwards Pods backing Services on a remote Kubernetes cluster to a matching set of DNS names and ports on the local workstation (in this case a Jupyter Notebook). Once kubefwd is running, connections to services such as http://elasticsearch:9200 are possible just as they are from within the remote cluster.

Create a new Python 3 Jupyter Notebook; copy and execute the following code examples within individual cells.

```
!pip install elasticsearch==7.6.0
```
```
from elasticsearch import Elasticsearch
import pandas as pd
from matplotlib import pyplot
```
Create an Elasticsearch client connected to the elasticsearch service
running in the Kubernetes Namespace data:

```
es = Elasticsearch(["elasticsearch:9200"])
```

Use the Elasticsearch client’s search function to query the index
pattern sentiment-*, and store the results in the variable response:
```
response = es.search(
    index="sentiment-*",
    body={
        "size": 10000,
        "query": {
            "range": {
                "Date": {
                    "gt": "now-1h"
                }
            }
        },
        "_source": [
            "Date",
            "polarity",
            "subjectivity" ],
    }
)
```
Map and transpose the response from Elasticsearch into Pandas
DataFrame:
```

df = pd.concat(map(pd.DataFrame.from_dict,
                   response['hits']['hits']),
               axis=1)['_source'].T
```

Convert the Date column to a Python Datetime data type:
```
datefmt = '%a, %d %b %Y %H:%M:%S GMT'
df['Date'] = pd.to_datetime(df['Date'], format=datefmt)
```

Assign the Date field to the DataFrame index and convert all numeric
values to floats:
```
df = df.set_index(['Date'])
df = df.astype(float)
```
Print the first five records:
```
df.head()
```
Sample Sentiment Analysis DataFrame rows

<img>

Finally, plot sentiment by calling the plot function of the DataFrame,
assigning polarity to the y axis :

```
df.plot(y=["polarity"], figsize=(13,5))
```


#### Kibana

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo1-DataProcessing-Serverless-ETL/pictures/Kibana.png" width="800">








