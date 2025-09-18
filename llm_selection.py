import numpy as np
import argparse

def main(args):
    if args.dataset == 'movielens-1m':
        features = ["user_id", "movie_id","genres","age","gender","occupation","title","timestamp", "zip"]

        gpt4o = np.load('selected_features_gpt-4o_movielens.npy').tolist()
        gpt4 = np.load('selected_features_gpt-4-turbo_movielens.npy').tolist()
        gpt35 = np.load('selected_features_gpt-3.5-turbo-instruct_movielens.npy').tolist()
    elif args.dataset == 'aliccp':
        features = ['109_14', '508', '702', '210', '206', '125', '110_14', '205', '129', '509', '127_14', '150_14', '207', '216', '124', '128', '853', '301', '126', '127', '101', '121', '122']
        gpt4o = np.load('selected_features_gpt-4o_aliccp.npy').tolist()
        gpt4 = np.load('selected_features_gpt-4-turbo_aliccp.npy').tolist()
        gpt35 = np.load('selected_features_gpt-3.5-turbo-instruct_aliccp.npy').tolist()
    elif args.dataset == 'kuairand-pure':
        features = ['user_id', 'video_id', 'time_ms', 'is_rand', 'tab', 'user_active_degree', 'is_lowactive_period', 'is_live_streamer', 'is_video_author', 'follow_user_num_x', 'follow_user_num_range', 'fans_user_num', 'fans_user_num_range', 'friend_user_num', 'friend_user_num_range', 'register_days', 'register_days_range', 'onehot_feat0', 'onehot_feat1', 'onehot_feat2', 'onehot_feat3', 'onehot_feat4', 'onehot_feat5', 'onehot_feat6', 'onehot_feat7', 'onehot_feat8', 'onehot_feat9', 'onehot_feat10', 'onehot_feat11', 'onehot_feat12', 'onehot_feat13', 'onehot_feat14', 'onehot_feat15', 'onehot_feat16', 'onehot_feat17', 'author_id', 'video_type', 'upload_type', 'visible_status', 'video_duration', 'server_width', 'server_height', 'music_id', 'music_type', 'tag', 'counts', 'show_cnt', 'show_user_num', 'play_cnt', 'play_user_num', 'play_duration', 'complete_play_cnt', 'complete_play_user_num', 'valid_play_cnt', 'valid_play_user_num', 'long_time_play_cnt', 'long_time_play_user_num', 'short_time_play_cnt', 'short_time_play_user_num', 'play_progress', 'comment_stay_duration', 'like_cnt', 'like_user_num', 'click_like_cnt', 'double_click_cnt', 'cancel_like_cnt', 'cancel_like_user_num', 'comment_cnt', 'comment_user_num', 'direct_comment_cnt', 'reply_comment_cnt', 'delete_comment_cnt', 'delete_comment_user_num', 'comment_like_cnt', 'comment_like_user_num', 'follow_cnt', 'follow_user_num_y', 'cancel_follow_cnt', 'cancel_follow_user_num', 'share_cnt', 'share_user_num', 'download_cnt', 'download_user_num', 'report_cnt', 'report_user_num', 'reduce_similar_cnt', 'reduce_similar_user_num', 'collect_cnt', 'collect_user_num', 'cancel_collect_cnt', 'cancel_collect_user_num', 'direct_comment_user_num', 'reply_comment_user_num', 'share_all_cnt', 'share_all_user_num', 'outsite_share_all_cnt']
        gpt4o = np.load('selected_features_gpt-4o_kuairand.npy').tolist()
        gpt4 = np.load('selected_features_gpt-4-turbo_kuairand.npy').tolist()
        gpt35 = np.load('selected_features_gpt-3.5-turbo-instruct_kuairand.npy').tolist()
        for lis in [gpt4o, gpt4, gpt35]:
            for ele in lis:
                if ele not in features:
                    lis.remove(ele)
        for lis in [gpt4o, gpt4, gpt35]:
            for ele in lis:
                if ele not in features:
                    lis.remove(ele)

    scores = []
    feature_num = len(features)

    for feature in features:
        tmp_score = 0
        for idx, lis in enumerate([gpt4o, gpt4, gpt35]):
            flag = lis.index(feature)
            tmp_score += (1 - flag * (1/feature_num)) * args.gate[idx]
        scores.append(tmp_score)

    sorted_features = sorted(zip(scores, features), reverse=True)
    sorted_features = [feature for score, feature in sorted_features]

    print(sorted_features)

    sorted_scores = [score for score, feature in sorted_features]
    ans = np.array([sorted_features, sorted_scores])
    np.save('feature_rank.npy', ans)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Selection')
    parser.add_argument('--dataset', type=str, default='kuairand-pure', help='Dataset name')
    parser.add_argument('--gate', type=float, nargs='+', default=[0.33, 0.33, 0.33], help='Weights for the models')
    args = parser.parse_args()
    main(args)
