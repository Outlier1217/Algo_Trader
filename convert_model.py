from stable_baselines3 import PPO

model = PPO.load("btc_hybrid_final_300k")
model.save("btc_hybrid_final_300k_vps")
