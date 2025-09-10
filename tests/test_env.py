def test_dummy_env():
    import vlarl_infra
    import gymnasium as gym
    
    env = gym.make("Dummy-v1")
    print("Env created:", env)
    obs, info = env.reset()
    print("Reset observation:", obs)
    action = env.unwrapped.fake_action()
    print("Fake action:", action)
    obs, reward, terminated, truncated, info = env.step(action)
    print("Step result:", obs, reward, terminated, truncated, info)
    
    
if __name__ == "__main__":
    test_dummy_env()