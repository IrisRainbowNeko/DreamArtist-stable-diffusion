@echo Off
TITLE Download Model Files
Pause
curl --create-dirs -A "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64)" -L "https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1" -o "%cd%"/experiments/pretrained_models/project.yaml --ssl-no-revoke
curl -A "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64)" -L "https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1" -o "%cd%"/experiments/pretrained_models/model.ckpt --ssl-no-revoke
