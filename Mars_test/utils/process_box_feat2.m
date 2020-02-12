function video_feat = process_box_feat2(box_feat, video_info, frame)

nVideo = size(video_info, 1);
video_feat = zeros(size(box_feat, 1), nVideo);
for n = 1:nVideo
    start_id = video_info(n, 1);
    end_id = video_info(n, 2);
    
    if frame ==1
        video_feat(:, n) =box_feat(:, start_id);
    else
        end_id = min(end_id, start_id + frame -1);
        feature_set = box_feat(:, start_id : end_id);
        video_feat(:, n) = mean(feature_set, 2); 
        %video_feat(:, n) = max(feature_set, [], 2);
    end
end

%%% normalize train and test features
sum_val = sqrt(sum(video_feat.^2));
for n = 1:size(video_feat, 1)
    video_feat(n, :) = video_feat(n, :)./sum_val;
end