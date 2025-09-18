from openai import OpenAI
import argparse
import numpy as np

def main(args):
    movielens_prompt = '''
    You are a professional researcher in recommender system and feature selection. Given a deep learning task and descriptions of some features, you can accurately select informative and effective features to help the deep learning model more accurately and robustly model the task, achieving optimal performance.
    Next, I will provide you with a description of a deep learning modeling task, descriptions of a series of features, as well as the features already selected and a set of candidate features. Please analyze the provided information and choose one feature from the candidate set that you think is crucial.
    Description of the deep learning modeling task: given user features, movie features, interaction features, the task is to predict whether the user will watch the movie.
    Descriptions of dataset:
    Field Name:	Description	value
    Sample num	The number of samples in the dataset	1000209
    Feature num	The number of features	9

    Descriptions of features:
    Interaction features:
    Field Name:	Description	Type	Example	unique values
    user_id	The ID of the user. UserIDs range between 1 and 6040.	int64	1	6040
    movie_id	The ID of the movie. MovieIDs range between 1 and 3952.	int64	661	3952
    timestamp	Timestamp of the interaction. Timestamp is represented in seconds since the epoch as returned by time(2)	int64	978300719	458455
    User features:
    Field Name	Description	Type	Example	unique values
    user_id	The ID of the user. UserIDs range between 1 and 6040.	int64	1	6040
    gender	Gender is denoted by a "M" for male and "F" for female	str	F	2
    age	Age is chosen from the following ranges:

        *  1:  "Under 18"
        * 18:  "18-24"
        * 25:  "25-34"
        * 35:  "35-44"
        * 45:  "45-49"
        * 50:  "50-55"
        * 56:  "56+"	int64	56	7
    Occupation	 Occupation is chosen from the following choices:

        *  0:  "other" or not specified
        *  1:  "academic/educator"
        *  2:  "artist"
        *  3:  "clerical/admin"
        *  4:  "college/grad student"
        *  5:  "customer service"
        *  6:  "doctor/health care"
        *  7:  "executive/managerial"
        *  8:  "farmer"
        *  9:  "homemaker"
        * 10:  "K-12 student"
        * 11:  "lawyer"
        * 12:  "programmer"
        * 13:  "retired"
        * 14:  "sales/marketing"
        * 15:  "scientist"
        * 16:  "self-employed"
        * 17:  "technician/engineer"
        * 18:  "tradesman/craftsman"
        * 19:  "unemployed"
        * 20:  "writer"	int64	16	21
    zip	zip code	int64	48067	3439
    Movie features:
    Field Name	Description	Type	Example	unique value
    movie_id	The ID of the movie. MovieIDs range between 1 and 3952.	int64	661	3952
    title	Movie titles. Titles are identical to titles provided by the IMDB (including
    year of release)	str	One Flew Over the Cuckoo's Nest (1975)	3706
    genres	Genres are pipe-separated and are selected from the following genres:

        * Action
        * Adventure
        * Animation
        * Children's
        * Comedy
        * Crime
        * Documentary
        * Drama
        * Fantasy
        * Film-Noir
        * Horror
        * Musical
        * Mystery
        * Romance
        * Sci-Fi
        * Thriller
        * War
        * Western	str	Adventure|Children's|Drama|Musical	301
    Features already selected: {selected_features}
    Candidate features: {candidate_features}
    Please select one feature from the candidate feature set that you think is most important for this task to add to the selected features. Suppose we will discrete the features simply based on their unique values. The feature you choose should, as much as possible, have the following characteristics: 
    1.They are informative.
    2.Independent of other selected features. 
    3.They are simple and easy for the model to understand.
    Your answer's last line should be the feature name of your selected feature, without any other characters.
    '''

    aliccp_prompt = """
    You are a professional researcher in recommender system and feature selection. Given a deep learning task and descriptions of some features, you can accurately select informative and effective features to help the deep learning model more accurately and robustly model the task, achieving optimal performance.
    Next, I will provide you with a description of a deep learning modeling task, descriptions of a series of features, as well as the features already selected and a set of candidate features. Please analyze the provided information and choose one feature from the candidate set that you think is crucial.
    Description of the deep learning modeling task: given user features, item features, interaction features, the task is to predict whether the user will click on the item.
    Descriptions of dataset:
    Field Name:	Description	value
    Sample num	The number of samples in the dataset	85316519
    Feature num	The number of features	23
    Descriptions of features:
    User features:
    All features are discretized.
    Field Name:	Description	Type	Example	unique values
    101	User ID	int64	24	238635
    109_14	User historical behaviors of category ID and count*.	int64	36	5853
    110_14	User historical behaviors of shop ID and count*.	int64	4	105622
    127_14	User historical behaviors of brand ID and count*.	int64	7	53843
    150_14	User historical behaviors of intention node ID and count*.	int64	12	31858
    121	Categorical ID of User Profile.	int64	6	98
    122	Categorical group ID of User Profile.	int64	96	14
    124	Users Gender ID.	int64	55	3
    125	Users Age ID.	int64	43	8
    126	Users Consumption Level Type I.	int64	28	4
    127	Users Consumption Level Type II.	int64	24	4
    128	Users Occupation: whether or not to work.	int64	36	3
    129	Users Geography Informations.	int64	14	5
    Item features:
    All features are discretized.
    Field Name:	Description	Type	Example	unique values
    205	Item ID.	int64	21	467298
    206	Category ID to which the item belongs to.	int64	4	6929
    207	Shop ID to which item belongs to.	int64	12	263942
    210	Intention node ID which the item belongs to.	int64	5	80232
    216	Brand ID of the item.	int64	36	106399
    Interaction features:
    All features are discretized.

    Field Name:	Description	Type	Example	unique values
    508	The combination of features with 109_14 and 206.	int64	27	5888
    509	The combination of features with 110_14 and 207.	int64	2	104830
    702	The combination of features with 127_14 and 216.	int64	4	51878
    853	The combination of features with 150_14 and 210.	int64	10	37148
    301	A categorical expression of position.	int64	3	3
    Features already selected: {selected_features}
    Candidate features: {candidate_features}
    Please select one feature from the candidate feature set that you think is most important for this task to add to the selected features. Suppose we will discrete the features simply based on their unique values. The feature you choose should, as much as possible, have the following characteristics: 
    1.They are informative.
    2.Independent of other selected features. 
    3.They are simple and easy for the model to understand.
    Your answer's last line should be the feature name of your selected feature, without any other characters.
    """

    kuairand_prompt = '''
    You are a professional researcher in recommender system and feature selection. Given a deep learning task and descriptions of some features, you can accurately select informative and effective features to help the deep learning model more accurately and robustly model the task, achieving optimal performance.
    Next, I will provide you with a description of a deep learning modeling task, descriptions of a series of features, as well as the features already selected and a set of candidate features. Please analyze the provided information and choose one feature from the candidate set that you think is crucial.
    Description of the deep learning modeling task: given user features, video features, interaction features, the task is to predict whether the user will click on the video.
    Descriptions of dataset:
    Field Name:	Description	value
    Sample num	The number of samples in the dataset	1,436,609
    Feature num	The number of features	102

    Descriptions of features:
    Interaction features:
    Field Name:	Description	Type	Example
    user_id	The ID of the user.	int64	17387
    video_id	The ID of the video.	int64	1123453
    date	The date of this interaction	int64	20220421
    hourmin	The time of this interaction (format: HHSS).	int64	400
    time_ms	The timestamp of this interaction in milliseconds.	int64	1650485801301
    profile_stay_time	The time that the user stayed in this author's profile.	int64	0
    comment_stay_time	The time that the user stayed in the comments section of this video	int64	0
    is_profile_enter	A binary feedback signal indicating the user enters the author profile	int64	0
    is_rand	A binary signal indicating if this video is generated by the random intervention (i.e., a random exposed video).	int64	0
    tab	indicating the scenario of this interaction, e.g., in the recommendation page or the main page of the App. In the range of [0,14].	int64	1
    User features:

    Field Name:	Description	Type	Example
    user_id	The ID of the user.	int64	25621
    user_active_degree	In the set of {'high_active', 'full_active', 'middle_active', 'UNKNOWN'}.	str	“full_active”
    is_lowactive_period	Is this user in its low active period	int64	0
    is_live_streamer	Is this user a live streamer?	int64	0
    is_video_author	Has this user uploaded any video?	int64	1
    follow_user_num_x	The number of users that this user follows.	int64	5
    follow_user_num_range	The range of the number of users that this user follows. In the set of {'0', '(0,10]', '(10,50]', '(100,150]', '(150,250]', '(250,500]', '(50,100]', '500+'}	str	“(0,10]”
    fans_user_num	The number of the fans of this user.	int64	312
    fans_user_num_range	The range of the number of fans of this user. In the set of {'0', '[1,10)', '[10,100)', '[100,1k)', '[1k,5k)', '[5k,1w)', '[1w,10w)'}	str	“[100,1k)”
    friend_user_num	The number of friends that this user has.	int64	0
    friend_user_num_range	The range of the number of friends that this user has. In the set of {'0', '[1,5)', '[5,30)', '[30,60)', '[60,120)', '[120,250)', '250+'}	str	“0”
    register_days	The days since this user has registered.	int64	3624
    register_days_range	The range of the registered days. In the set of {'15-30', '31-60', '61-90', '91-180', '181-365', '366-730', '730+'}.	str	“730+”
    onehot_feat0	An encrypted feature of the user. Each value indicates the position of “1” in the one-hot vector. Range: {0,1}	int64	1
    onehot_feat1	An encrypted feature. Range: {0, 1, …, 6}	int64	2
    onehot_feat2	An encrypted feature. Range: {0, 1, …, 49}	int64	2
    onehot_feat3	An encrypted feature. Range: {0, 1, …, 1470}	int64	1153
    onehot_feat4	An encrypted feature. Range: {0, 1, …, 14}	int64	4
    onehot_feat5	An encrypted feature. Range: {0, 1, …, 33}	int64	0
    onehot_feat6	An encrypted feature. Range: {0, 1, 2}	int64	0
    onehot_feat7	An encrypted feature. Range: {0, 1, …, 117}	int64	31
    onehot_feat8	An encrypted feature. Range: {0, 1, …, 453}	int64	354
    onehot_feat9	An encrypted feature. Range: {0, 1, …, 6}	int64	3
    onehot_feat10	An encrypted feature. Range: {0, 1, …, 4}	int64	3
    onehot_feat11	An encrypted feature. Range: {0, 1, …, 4}	int64	2
    onehot_feat12	An encrypted feature. Range: {0, 1}	int64	1
    onehot_feat13	An encrypted feature. Range: {0, 1}	int64	0
    onehot_feat14	An encrypted feature. Range: {0, 1}	int64	0
    onehot_feat15	An encrypted feature. Range: {0, 1}	int64	0
    onehot_feat16	An encrypted feature. Range: {0, 1}	int64	0
    onehot_feat17	An encrypted feature. Range: {0, 1}	int64	0
    Video features

    Field Name:	Description	Type	Example
    video_id	The ID of the video.	int64	3784
    author_id	The ID of the author of this video. In the range of [0, 8839734]	int64	441
    video_type	Type of this video (NORMAL or AD).	str	“NORMAL”
    upload_dt	Upload date of this video.	str	“2020-07-08”
    upload_type	The upload type of this video.	str	“ShortImport”
    visible_status	The visible state of this video on the APP now.	int	1
    video_duration	The time duration of this duration (in milliseconds).	Int64	17200.0
    server_width	The width of this video on the server.	int64	720
    server_height	The height of this video on the server.	int64	1280
    music_id	Background music ID of this video.	int64	989206467
    music_type	Background music type of this video.	int64	4
    tag	A list of key categories (labels) of this video.	str	“12,65”
    Video static features

    Field Name:	Description	Type	Example
    video_id	The ID of the video.	int64	9288071
    counts	The number of statistics.	int64	66
    show_cnt	The number of shows of this video (averaged on each day and each scenario over one month. This applies to all the following fields)	float64	75.212
    show_user_num	The number of users who received the recommendation of this video.	float64	66.985
    play_cnt	The number of plays.	float64	9.409
    play_user_num	The number of users who play this video.	float64	8.121
    play_duration	The total time duration of playing this video (in milliseconds).	float64	93700.333
    complete_play_cnt	The number of complete plays. complete play: finishing playing the whole video, i.e., #(play_duration >= video_duration).	float64	0.182
    complete_play_user_num	The number of users who perform the complete play.	float64	0.182
    valid_play_cnt	valid play: play_duration >= video_duration if video_duration <= 7s, or play_duration > 7 if video_duration > 7s.	float64	3.545
    valid_play_user_num	The number of users who perform the complete play.	float64	3.136
    long_time_play_cnt	long time play: play_duration >= video_duration if video_duration <= 18s, or play_duration >=18 if video_duration > 18s.	float64	1.909
    long_time_play_user_num	The number of users who perform the long time play.	float64	1.848
    short_time_play_cnt	short time play: play_duration < min(3s, video_duration).	float64	5.015
    short_time_play_user_num	The number of users who perform the short time play.	float64	4.545
    play_progress	The average video playing ratio (=play_duration/video_duration)	float64	0.016
    comment_stay_duration	Total time of staying in the comments section	float64	2302.712
    like_cnt	Total likes	float64	0.303
    like_user_num	The number of users who hit the “like” button.	float64	0.303
    click_like_cnt	The number of the “like” resulted from double click	float64	0.030
    double_click_cnt	The number of users who double-click the video.	float64	0.273
    cancel_like_cnt	The number of likes that are canceled by users.	float64	0.485
    cancel_like_user_num	The number of users who cancel their likes.	float64	0.485
    comment_cnt	The number of comments within this day.	float64	0.015
    comment_user_num	The number of users who comment on this video.	float64	0.015
    direct_comment_cnt	The number of direct comments (depth=1).	float64	0.015
    reply_comment_cnt	The number of reply comments (depth>1).	float64	0.000
    delete_comment_cnt	The number of deleted comments.	float64	0.015
    delete_comment_user_num	The number of users who delete their comments.	float64	0.015
    comment_like_cnt	The number of comment likes.	float64	0.000
    comment_like_user_num	The number of users who like the comments.	float64	0.000
    follow_cnt	The number of increased follows from this video.	float64	0.000
    follow_user_num_y	The number of users who follow the author of this video due to this video.	float64	0.000
    cancel_follow_cnt	The number of decreased follows from this video.	float64	0.000
    cancel_follow_user_num	The number of users who cancel their following of the author of this video due to this video.	float64	0.000
    share_cnt	The times of successfully sharing this video.	float64	0.000
    share_user_num	The number of users who succeed to share this video.	float64	0.000
    download_cnt	The times of downloading this video.	float64	0.030
    download_user_num	The number of users who download this video.	float64	0.030
    report_cnt	The times of reporting this video.	float64	0.000
    report_user_num	The number of users who report this video.	float64	0.000
    reduce_similar_cnt	The times of reducing similar content of this video.	float64	0.015
    reduce_similar_user_num	The number of users who choose to reduce similar content of this video.	float64	0.015
    collect_cnt	The times of adding this video to favorite videos.	float64	0.061
    collect_user_num	The number of users who add this video to their favorite videos.	float64	0.061
    cancel_collect_cnt	The times of removing this video from favorite videos.	float64	0.091
    cancel_collect_user_num	The number of users who remove this video from their favorite videos	float64	0.091
    direct_comment_user_num	The number of users who write comments directly under this video (level=1).	float64	0.015
    reply_comment_user_num	The number of users who reply the existing comments (level>1).	float64	0.000
    share_all_cnt	The times of sharing this video (no need to be successful).	float64	0.015
    share_all_user_num	The number of users who share this video (no need to be successful).	float64	0.015
    outsite_share_all_cnt	The times of sharing this video outside Kuaishou App.	float64	0.000

    Features already selected: {selected_features}
    Candidate features: {candidate_features}
    Please select one feature from the candidate feature set that you think is most important for this task to add to the selected features. Suppose we will discrete the features simply based on their unique values. The feature you choose should, as much as possible, have the following characteristics: 
    1.They are informative. 
    2.Independent of other selected features. 
    3.They are simple and easy for the model to understand.
    Your answer's last line should be the feature name of your selected feature, without any other characters. The feature in features already selected should not be selected again.
    '''

    extract_prompt = "{answer} \n Please output the selected feature without any other information."
    extract_prompts = [
        {"role": "user", "content": ""}
    ]

    if args.dataset == 'movielens':
        features = ["user_id", "movie_id","genres","age","gender","occupation","title","timestamp", "zip"]
        # from long to short, to make sure the longer feature is selected first
        features = sorted(features, key=len, reverse=True)
        prompts = [
            {"role": "user", "content": movielens_prompt}
        ]
    elif args.dataset == 'aliccp':
        features = ['109_14', '508', '702', '210', '206', '125', '110_14', '205', '129', '509', '127_14', '150_14', '207', '216', '124', '128', '853', '301', '126', '127', '101', '121', '122']
        features = sorted(features, key=len, reverse=True)
        prompts = [
            {"role": "user", "content": aliccp_prompt}
        ]
    elif args.dataset == 'kuairand':
        features = ["onehot_feat0", "onehot_feat1","onehot_feat2", "onehot_feat3", "onehot_feat4", "onehot_feat5", "onehot_feat6", "onehot_feat7","onehot_feat8", "onehot_feat9", "onehot_feat10", "onehot_feat11", "onehot_feat12", "onehot_feat13","onehot_feat14", "onehot_feat15", "onehot_feat16", "onehot_feat17", "profile_stay_time", "play_duration", "comment_stay_duration", "video_type", "user_active_degree", "is_video_author", "video_duration", "tab", "follow_user_num_x", "follow_user_num_y", "comment_stay_time", "hourmin", "is_profile_enter", "register_days", "like_user_num", "fans_user_num", "music_id", "follow_cnt", "author_id", "is_lowactive_period", "time_ms", "upload_dt", "visible_status", "play_progress", "upload_type", "server_width", "tag", "music_type", "show_cnt", "date", "is_rand", "server_height", "valid_play_cnt", "play_user_num", "video_id", "user_id", "play_cnt", "long_time_play_user_num", "is_live_streamer", "show_user_num", "short_time_play_cnt", "complete_play_cnt", "valid_play_user_num", "like_cnt", "complete_play_user_num", "double_click_cnt", "comment_cnt", "comment_user_num", "click_like_cnt", "share_cnt", "collect_cnt", "share_all_cnt", "cancel_follow_user_num", "short_time_play_user_num", "friend_user_num", "share_user_num", "cancel_like_user_num", "comment_like_user_num", "direct_comment_cnt", "long_time_play_cnt", "download_user_num", "download_cnt", "cancel_like_cnt","report_cnt", "fans_user_num_range", "reply_comment_cnt", "reduce_similar_cnt", "delete_comment_cnt", "register_days_range", "report_user_num", "reduce_similar_user_num", "cancel_follow_cnt", "delete_comment_user_num", "share_all_user_num", "follow_user_num_range","collect_user_num", "direct_comment_user_num","comment_like_cnt","reply_comment_user_num", "counts", "outsite_share_all_cnt", "cancel_collect_user_num", "cancel_collect_cnt","friend_user_num_range"]
        features = sorted(features, key=len, reverse=True)
        prompts = [
            {"role": "user", "content": kuairand_prompt}
        ]
    client = OpenAI(
            api_key = "",
            base_url = ""
        )

    # initialize selected features and candidate features
    if args.dataset == 'movielens':
        selected_features = ["user_id", "movie_id"]
        candidate_features = features.copy()
    elif args.dataset == 'aliccp':
        selected_features = ['101', '205']
        candidate_features = features.copy()
    elif args.dataset == 'kuairand':
        selected_features = ['user_id', 'video_id']
        candidate_features = features.copy()
    for feature in selected_features:
        candidate_features.remove(feature)


    while len(candidate_features) > 0:
        if args.dataset == 'movielens':
            prompts[0]['content'] = movielens_prompt.format(selected_features = selected_features, candidate_features = candidate_features)
        elif args.dataset == 'aliccp':
            prompts[0]['content'] = aliccp_prompt.format(selected_features = selected_features, candidate_features = candidate_features)
        elif args.dataset == 'kuairand':
            tmp_prompt = kuairand_prompt
            tmp_prompt = tmp_prompt.replace("{selected_features}",str(selected_features))
            tmp_prompt = tmp_prompt.replace("{candidate_features}", str(candidate_features))
            prompts[0]['content'] = tmp_prompt
        response = client.chat.completions.create(
            model = args.model,
            messages = prompts,
            max_tokens= 1000,
            temperature = 0.7,
            n = 1,
            stop = None
        )
        ans = response.choices[0].message.content
        print(ans)
        # get the selected feature
        last_line = ans.split("\n")[-1]
        # last_line = response.choices[0].message.content.lower()
        print(last_line)
        for feature in candidate_features:
            if feature in last_line:
                print('select ', feature)
                selected_features.append(feature)
                candidate_features.remove(feature)
                print(selected_features)
                break
    
    print('final ranking: ', selected_features)
    # save npy
    np.save('../selected_features_{}_{}.npy'.format(args.model, args.dataset), selected_features)
    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4-turbo", help="The model to use for the completion.") # gpt-3.5-turbo-instruct, gpt-4o, gemini-1.5-pro, gpt-4-turbo
    parser.add_argument("--dataset", type=str, default="movielens", help="The dataset to use for the prompt.")
    args = parser.parse_args()
    main(args)