select
mrm.feelyt_id
, user_age
, user_gender
, occasion_relax
, occasion_friends
, occasion_family
, occasion_date
, occasion_time_burning
, occasion_girls_night
, occasion_boys_night
, occasion_trouble
, atmosphere_food_for_thought
, atmosphere_funny
, atmosphere_action
, atmosphere_emotional
, atmosphere_romantic
, atmosphere_dark
, atmosphere_brutal
, atmosphere_thrilling
, feeling_happy
, feeling_strained
, feeling_stressed
, feeling_unsatisfied
, feeling_sad
, feeling_listless
, feeling_agitated
, feeling_euphoric
, description_de
, description_en
, fsk_rating
, imdb_id
, tmdb_id
, amg_id
, rotten_id
, iva_id
, title
, year
, runtime

 from
movie_rating_manual as mrm

LEFT JOIN movie_info as mi
ON mrm.feelyt_id = mi.feelyt_id

LEFT JOIN movie_identifier as mid
ON mid.feelyt_id = mrm.feelyt_id
