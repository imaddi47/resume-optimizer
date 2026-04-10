import pytest
from app.models.ai_config import UserAIConfig, AIProvider


@pytest.mark.asyncio
async def test_create_ai_config(db_session, test_user):
    config = UserAIConfig(
        user_id=test_user.id,
        provider=AIProvider.OPENAI,
        api_key_encrypted="encrypted-data",
        model_id="gpt-4o",
    )
    db_session.add(config)
    await db_session.commit()
    await db_session.refresh(config)

    assert config.id is not None
    assert config.user_id == test_user.id
    assert config.provider == AIProvider.OPENAI
    assert config.model_id == "gpt-4o"


@pytest.mark.asyncio
async def test_ai_config_unique_per_user(db_session, test_user):
    config1 = UserAIConfig(
        user_id=test_user.id,
        provider=AIProvider.OPENAI,
        api_key_encrypted="encrypted-1",
        model_id="gpt-4o",
    )
    db_session.add(config1)
    await db_session.commit()

    config2 = UserAIConfig(
        user_id=test_user.id,
        provider=AIProvider.ANTHROPIC,
        api_key_encrypted="encrypted-2",
        model_id="claude-sonnet-4-6",
    )
    db_session.add(config2)
    with pytest.raises(Exception):  # IntegrityError — unique constraint
        await db_session.commit()
